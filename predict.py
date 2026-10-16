import matplotlib.pyplot as plt
import warnings
import argparse, os, tqdm
import torchcrepe, torch, librosa, soundfile
import pandas as pd, numpy as np
from scipy import signal
warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser()
parser.add_argument('input', type=str, help="Directory with sound files to process, or a single file to process")
parser.add_argument('--high_pass', type=float, help="High pass cut-off frequency", default=None)
parser.add_argument('--model_path', type=str, help="Path of model weights", default=os.path.join(os.path.dirname(__file__), 'model_all.pth'))
parser.add_argument('--compress', type=float, help="Compression factor used to shift frequencies into CREPE's range [32Hz; 2kHz]. \
    Frequencies are divided by the given factor by artificially changing the sampling rate (slowing down / speeding up the signal).", default=1)
parser.add_argument('--use_channel', type=int, help='ID of the channel to use (starting from 0, if not specified, will use a mix of all)', default=None)
# Sample rate - controls how frequently CREPE makes an F0 prediction (normally 16000 Hz)
#   --step 0.01 gives one prediction every 10 ms -> 100 predictions per second
parser.add_argument('--step', type=float, help="Step used between each prediction (in seconds, default is 0.016)", default=256 / torchcrepe.SAMPLE_RATE)
parser.add_argument('--decoder', choices=['argmax', 'weighted_argmax', 'viterbi'], help="Decoder used to postprocess predictions", default='weighted_argmax')

# Does not create a PNG spectogram
parser.add_argument('--no_print', action='store_true', help="Skip printing spectrograms with overlaid F0 predictions to assess their quality")
parser.add_argument('--no_characterisation', action='store_true', help="Skip the computation of vocalisation characteristics (harmonicity, salience, and SHR)")
parser.add_argument('--threshold', type=float, help="Confidence threshold used when printing F0 predictions on spectrograms ", default=0.1)
parser.add_argument('--NFFT', type=int, help="Window size used for the spectrum computation (for printing F0 predictions and computing vocalisation characteristics)", default=1024)
args = parser.parse_args()

# Initialisations
# Checks if you have CUDA-compatible GPU, otherwise uses the CPU (affecting processing speed)
device, batch_size = ('cuda', 64) if torch.cuda.is_available() else ('cpu', 1)
# .eval() = evaluation mode, to(device) moves it either to CPU or GPU
model = torchcrepe.Crepe('full').eval().to(device)
# Load trained model weights
model.load_state_dict(torch.load(args.model_path, map_location=device))
decoder = torchcrepe.decode.__dict__[args.decoder] # Selects decoder

# SAMPLING  
# The FFT is used later for the additional vocalisation measurements, for spectral analysis (SHR/harmonicity calcs)
FS, nfft = int(torchcrepe.SAMPLE_RATE * args.compress), args.NFFT
Hz2bin = lambda f: np.floor(f / FS * nfft).astype(int)
# Hann window - used to convert signal from seeming like it runs infinitely to gradual up and down from 0 amplitude
# If not using Hann window, spectual leakage -> If the beginning and end don't match smoothly, you get an artificial discontinuity:
hann = np.hanning(nfft)
get_spectrum = lambda x : np.abs(np.fft.rfft(hann * x))
# Creates high pass filter if --high_pass was supplied - cuts off lower frequencies
high_pass = signal.butter(5, 2*args.high_pass/FS, 'hp', output='sos') if args.high_pass else None

# FINDS AUDIO FILE/DIRECTORY
if os.path.isdir(args.input):
    files = [os.path.join(args.input, fn) for fn in os.listdir(args.input) if fn.split('.')[-1].upper() in soundfile._formats]
elif os.path.isfile(args.input) and args.input.split('.')[-1].upper() in soundfile._formats:
    files = [args.input]
else:
    files = []

if len(files) == 0:
    raise Exception(f"The given input argument {args.input} is either not a valid directory/file path, or it doesn't contain any sound file of the supported formats (please refer to pysoundfile documentation)")

# prints the effective F0 range
print(f'With the current compression factor of {args.compress}, the model\'s F0 estimations range from {32*args.compress}Hz to {2000*args.compress}Hz')

for ifile, filepath in enumerate(files):
    try:
        # Load the audio - IMPORTANT: librosa resamples file to FS = 16000 Hz
        sig, fs = librosa.load(filepath, sr=FS, mono=args.use_channel is None)
        if sig.ndim > 1:
            sig = sig[args.use_channel] # if multi channel
    except Exception as e:
        # Skip failing file paths
        print(f'Failed to load {filepath}')
        print(f'Error: {e}')
        continue
    if args.high_pass:
        # Apply high pass if requested
        sig = signal.sosfilt(high_pass, sig).astype("float32")

    # Split audio into frames
    # Note: this is where the prediction sampling interval is actually used
    # hop_length determines how far apart successive
    # prediction windows are.
    #
    # Default:
    #
    # args.step = 0.016
    # compress = 1
    # SAMPLE_RATE = 16000
    #
    # hop_length =
    # 0.016 * 1 * 16000
    # = 256 samples
    #
    # 256 samples / 16000 samples/sec
    # = 0.016 seconds
    # = 16 ms
    #
    # So CREPE makes a prediction every 16 ms.
    generator = torchcrepe.core.preprocess(torch.tensor(sig).unsqueeze(0), torchcrepe.SAMPLE_RATE, \
        hop_length=int(args.step * args.compress * torchcrepe.SAMPLE_RATE), batch_size=batch_size, device=device)
    size = int(1 + len(sig) // (args.step * args.compress * torchcrepe.SAMPLE_RATE)) // batch_size
    # RUN CREPE on every audio frame
    with torch.inference_mode():
        preds = torch.vstack([model(frames).cpu() for frames in tqdm.tqdm(generator, desc=f'{ifile+1}/{len(files)}: {filepath.split("/")[-1]}', total=size, leave=False)]).T.unsqueeze(0)
        # Convert raw CREPE output into one F0 value for each prediction time
        # then multiply by compress to restore the original frequency scale
        f0s = (torchcrepe.core.postprocess(preds, decoder=decoder) * args.compress).squeeze()
    # Take highest probability for every prediction
    confidence = preds.max(axis=1)[0].squeeze()
    
    # CREATE time values associated with each F0 prediction
    # DEFAULT:
        #
        # step = 0.016
        # compress = 1
        # sample rate = 16000
        #
        # therefore:
        #
        # time = 0.000
        #        0.016
        #        0.032
        #        0.048
        #        0.064
        #        ...
        #
        # So you get one F0 estimate every 16 ms.
    time = np.arange(0, len(sig)+1, int(args.step * args.compress * torchcrepe.SAMPLE_RATE)) / fs

    # CREATE THE OUTPUT TABLE of:
        # time        = time of prediction in seconds
        # f0          = estimated fundamental frequency in Hz
        # confidence  = CREPE confidence
    df = pd.DataFrame({'time':time, 'f0':f0s, 'confidence':confidence})
    
    # Vocalisation characterisation (harmonicity, salience, SHR)
    # SHR = Subharmonic to harmonic ratio (how many subharmonic frequencies are there below the harmonic freq)
    # A higher SHR therefore indicates relatively more subharmonic energy.
    if not args.no_characterisation:
        # If user wants characterisations (--no_characterisation was NOT supplied)
        spectrums = np.vstack([get_spectrum(sig[t - nfft//2 : t + nfft//2]) for t in (df.time.sample(min(len(df), 100)) * fs).round().astype(int) if t > nfft/2 and t < len(sig)-nfft/2]).T
        
        # Estimate median and standard deviation of spectral noise
        mednoise, stdnoise = np.median(spectrums, axis=1), np.std(spectrums, axis=1)
        # Finds the upper frequency limit when calculating harmonicity and SHR
        # Uses whatever is smaller - Nyquist frequency or 5x the highest detected F0 (5 being the practical cutoff as harmonics can go on forever)
        # the maximum frequency you can represent is the Nyquist frequency, so min needs to be taken
        shr_ceil = min(fs/2, df.f0.max() * 5)
        # Loop through every F0 prediction
        for irow, row in tqdm.tqdm(df.iterrows(), desc=f'{ifile+1}/{len(files)}: {filepath.split("/")[-1]} - voc. characterisation', total=len(df), leave=False):
            if FS*row.time < nfft//2 or FS*row.time > len(sig) - nfft//2:
                continue
            # Skip points where not enough audio before/after the point for an FFT window
            # FFT window = small section of the recording 
            # Used to know what frequencies there are and roughly WHERE (if we FFT windowed the whole signal - wouldn't know where)
            spec = get_spectrum(sig[int(FS * row.time) - nfft//2 : int(FS * row.time) + nfft//2])
            # Normalise spectrum relative to estimated noise
            spec = np.clip((spec-mednoise)/stdnoise, 1e-12, 1e3)
            f0 = row.f0
            
            df.loc[irow, 'harmonicity'] = sum(spec[Hz2bin(np.arange(f0*2, shr_ceil, f0))]) /  sum(spec[Hz2bin(np.arange(f0, shr_ceil, f0))]) if f0 *2 < fs / 2 else None
            df.loc[irow, 'salience'] = sum(spec[Hz2bin(f0*2**(-1/12)):Hz2bin(f0*2**(1/12))+1]) / sum(spec[Hz2bin(f0*2**(-6/12)):Hz2bin(f0*2**(6/12))+1])
            df.loc[irow, 'SHR'] = sum(spec[Hz2bin(np.arange(f0, shr_ceil, f0)-f0/2)]) / sum(spec[Hz2bin(np.arange(f0, shr_ceil, f0))]) if f0 < fs/2 else None

    # SAVE THE CSV to original audio file path with the extension "_f0.csv"
    df.to_csv(filepath.rsplit('.',1)[0]+'_f0.csv', index=False)
    # Plot F0 predictions over spectrograms
    if not args.no_print and len(sig)/fs <= 60:
        mask = confidence > args.threshold
        try:
            if mask.any():
                plt.figure(figsize=(max(4.8, 6.4*time[-1]/3*args.compress), 4.8))
                plt.specgram(sig, Fs=fs, NFFT=nfft, noverlap=nfft-nfft//8, cmap='inferno', vmin=-150)
                plt.scatter(time[mask], f0s[mask], c=confidence[mask], s=5, cmap='cool')
                plt.xlim(0, len(sig)/fs)
                plt.ylim(0, f0s[mask].max() * 1.5)
                plt.colorbar(label="Confidence")
                plt.xlabel('Time (sec)')
                plt.ylabel('Frequency (Hz)')
                plt.tight_layout()
                plt.savefig(filepath.rsplit('.',1)[0]+'_f0.png')
                plt.close()
            else:
                print(f'With the chosen confidence threshold {args.threshold}, no section was detected as voiced for {filepath}')
        except:
            print(f'Failed to create the spectrogram figure for {filepath}, but results are still saved in the .csv table')
