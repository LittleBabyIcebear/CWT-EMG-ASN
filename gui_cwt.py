import pandas as pd 
import plotly.graph_objects as go
import streamlit as st  
from numba import jit
from plotly.subplots import make_subplots
import math 
import numpy as np
import pywt
import wfdb

@jit(nopython=True)
def cwt(coloumncount, rowcount, a, da, dt, f0, y):
    w0 = 2 * np.pi * f0
    Ndata = len(y)
    pi = np.pi
    db = (Ndata - 1) * dt / coloumncount
    cwtre = np.zeros((coloumncount, rowcount))
    cwtim = np.zeros((coloumncount, rowcount))
    cwt = np.zeros((coloumncount, rowcount))

    for i in range(rowcount):
        b = 0.0
        for j in range(coloumncount):
            t = 0.0
            cwtre_sum = 0.0
            cwtim_sum = 0.0
            for k in range(Ndata):
                rem = (1 / np.sqrt(a)) * (1 / np.power(pi, 0.25)) * np.exp(-((t - b) / a) ** 2 / 2.0) * np.cos(w0 * (t - b) / a)
                imm = (1 / np.sqrt(a)) * (-1 / np.power(pi, 0.25)) * np.exp(-((t - b) / a) ** 2 / 2.0) * np.sin(w0 * (t - b) / a)
                cwtre_sum += y[k] * rem
                cwtim_sum += y[k] * imm
                t += dt

            cwtre[j, i] = cwtre_sum
            cwtim[j, i] = cwtim_sum
            cwt[j, i] = np.sqrt(cwtre[j, i] ** 2 + cwtim[j, i] ** 2)
            b += db

        a += da
    return cwt


def create_plotly_figure(title, xaxis, yaxis, time, signal, name, mode):
    # Create a Plotly figure
    fig = go.Figure()
    color= ['blue', 'red', 'green']
    for i in range(len(signal)):
        fig.add_trace(go.Scatter(x=time, y=signal[i], mode=mode, name=name[i], line=dict(color=color[i])))

    # Update layout
    fig.update_layout(
        title=title,
        xaxis_title=xaxis,
        yaxis_title=yaxis,
        width=800,
        height=500,
        xaxis=dict(showline=True, showgrid=True),
        yaxis=dict(showline=True, showgrid=True)
    )

    st.plotly_chart(fig)

def normalize_signal(signal, min_value=-100, max_value=100):
    signal_min = signal.min()
    signal_max = signal.max()
    # Scale the signal to the range [-100, 100]
    normalized_signal = (signal - signal_min) / (signal_max - signal_min) * (max_value - min_value) + min_value
    return normalized_signal

def butterworth_lowpass_filter(signal, cutoff_frequency, sampling_period, order):
    y = np.zeros(len(signal)) 
    omega_c = 2 * np.pi * cutoff_frequency
    omega_c_squared = omega_c * omega_c
    sampling_period_squared = sampling_period * sampling_period
    if order == 1:
        for n in range(len(signal)):
            if n == 0:
                y[n] = (omega_c * signal[n]) / ((2 / sampling_period) + omega_c)
            else:
                y[n] = (((2 / sampling_period) - omega_c) * y[n-1] + omega_c * signal[n] + omega_c * signal[n-1]) / ((2 / sampling_period) + omega_c)
    elif order == 2:
        y[0] = (omega_c * signal[0]) / ((2 / sampling_period) + omega_c)
        y[1] = (((2 / sampling_period) - omega_c) * y[0] + omega_c * signal[1] + omega_c * signal[0]) / ((2 / sampling_period) + omega_c)
        for n in range(2, len(signal)):
            y[n] = (((8 / sampling_period_squared) - 2 * omega_c_squared) * y[n-1]
                    - ((4 / sampling_period_squared) - (2 * np.sqrt(2) * omega_c / sampling_period) + omega_c_squared) * y[n-2]
                    + omega_c_squared * signal[n]
                    + 2 * omega_c_squared * signal[n-1]
                    + omega_c_squared * signal[n-2]) / ((4 / sampling_period_squared) + (2 * np.sqrt(2) * omega_c / sampling_period) + omega_c_squared)
    return y

def butterworth_highpass_filter(signal, cutoff_frequency, sampling_period, order):
    y = np.zeros(len(signal))  # Initialize the output signal
    omega_c = 2 * np.pi * cutoff_frequency
    omega_c_squared = omega_c * omega_c
    sampling_period_squared = sampling_period * sampling_period

    if order == 1:
        for n in range(len(signal)):
            if n == 0:
                y[n] = (omega_c * signal[n]) / ((2 / sampling_period) + omega_c)
            else:
                y[n] = (((2 / sampling_period) - omega_c) * y[n-1] + (2 / sampling_period) * signal[n] + (2 / sampling_period) * signal[n-1]) / ((2 / sampling_period) + omega_c)

    elif order == 2:
        y[0] = (omega_c * signal[0]) / ((2 / sampling_period) + omega_c)
        y[1] = (((2 / sampling_period) - omega_c) * y[0] + (2 / sampling_period) * signal[1] + (2 / sampling_period) * signal[0]) / ((2 / sampling_period) + omega_c)

        for n in range(2, len(signal)):
            y[n] = ((4 / sampling_period_squared) * signal[n] - (8 / sampling_period_squared) * signal[n-1] + (4 / sampling_period_squared) * signal[n-2]
                    - (2 * omega_c - (8 / sampling_period_squared)) * y[n-1]
                    - (omega_c - (2 * np.sqrt(2) * omega_c / sampling_period) + (4 / sampling_period_squared)) * y[n-2]) / (omega_c + 2 * np.sqrt(2) * omega_c / sampling_period + (4 / sampling_period_squared))    
    return y

def butterworth_bandpass_filter(signal, low_cutoff, high_cutoff, sampling_period, order):
    highpassed_signal = butterworth_highpass_filter(signal, low_cutoff, sampling_period, order)
    bandpassed_signal = butterworth_lowpass_filter(highpassed_signal, high_cutoff, sampling_period, order)
    return bandpassed_signal

def denoise_dwt(signal, level, threshold):
    # Perform wavelet decomposition using db4
    coeffs = pywt.wavedec(signal, 'db4', level=level)
    
    # Calculate noise standard deviation using robust median estimator
    sigma = np.median(np.abs(coeffs[-1])) / 0.6745
    
    # Initialize empty list for modified coefficients
    denoised_coeffs = []
    
    threshold_factor = threshold
    
    # Process each decomposition level with adaptive thresholding
    for i, coeff in enumerate(coeffs):
        if i == 0:  # Skip approximation coefficients
            denoised_coeffs.append(coeff)
            continue
            
        # Calculate level-dependent threshold with higher base threshold
        N = len(signal)
        level_factor = 1.5 / np.sqrt(2**(i))  # Increased level factor
        threshold = sigma * threshold_factor * np.sqrt(np.log(N)) * level_factor
        
        # Apply more aggressive thresholding
        coeff_abs = np.abs(coeff)
        sign = np.sign(coeff)
        
        # Increased threshold multiplier for classification
        is_small = coeff_abs <= 2.5 * threshold  # Increased from 2.0
        is_large = coeff_abs > 2.5 * threshold
        
        modified_coeff = np.zeros_like(coeff)
        # Only keep very large coefficients
        modified_coeff[is_large] = coeff[is_large]
        # More aggressive soft thresholding for small coefficients
        modified_coeff[is_small] = sign[is_small] * (coeff_abs[is_small] - 1.5 * threshold) * (coeff_abs[is_small] > 1.5 * threshold)
        
        denoised_coeffs.append(modified_coeff)
    
    # Reconstruct the signal using db4
    denoised_signal = pywt.waverec(denoised_coeffs, 'db4')
    
    # Ensure the output length matches input length
    if len(denoised_signal) > len(signal):
        denoised_signal = denoised_signal[:len(signal)]
    elif len(denoised_signal) < len(signal):
        denoised_signal = np.pad(denoised_signal, (0, len(signal) - len(denoised_signal)), 'edge')
    
    return denoised_signal


def plot_thresholded_cwt_with_boundaries(Z, threshold_coef=0.1):
    rowcount, colcount = Z.shape
    
    # Cari nilai maksimum dan minimum dari Z
    Z_max = np.max(Z)
    Z_min = np.min(Z)
    
    # Hitung threshold berdasarkan koefisien thresholding
    threshold = Z_min + threshold_coef * Z_max
    
    # Buat matriks thresholded Z (binary)
    Z_tr = np.where(Z > threshold, 1, 0)
    
    # Buat variabel anotate untuk menandai area threshold
    anotate = np.zeros(colcount)
    for x in range(colcount):
        if np.any(Z_tr[:, x] == 1):  
            anotate[x] = 1
    
    # Menghapus noise: jika ada nilai 1 tunggal di antara nilai 0, ubah menjadi 0
    for i in range(1, len(anotate) - 1):
        if anotate[i] == 1 and anotate[i - 1] == 0 and anotate[i + 1] == 0:
            anotate[i] = 0
    
    # Identifikasi batas antara area 0 dan 1
    boundaries = []
    for i in range(1, len(anotate)):
        if anotate[i] != anotate[i - 1]:  # Jika ada perubahan dari 0 ke 1 atau sebaliknya
            boundaries.append(i)

    # Buat subplot dengan 2 heatmap
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=('Original CWT', 'Thresholded CWT with Boundaries'),
        horizontal_spacing=0.1
    )
    
    # Plot original CWT
    fig.add_trace(
        go.Heatmap(
            z=Z,
            colorscale='Jet',
            showscale=True,
            colorbar=dict(x=0.45, title='Magnitude')
        ),
        row=1, col=1
    )
    
    # Plot thresholded CWT
    fig.add_trace(
        go.Heatmap(
            z=Z_tr,
            colorscale='Jet',
            showscale=True,
            colorbar=dict(x=1.0, title='Binary (0/1)')
        ),
        row=1, col=2
    )
    
    # Tambahkan garis batas pada plot thresholded CWT
    for boundary in boundaries:
        fig.add_shape(
            type="line",
            x0=boundary, y0=0, x1=boundary, y1=rowcount,
            line=dict(color="red", width=2),
            xref="x2", yref="y2"  # Mengacu ke subplot kedua
        )

    # Update layout
    fig.update_layout(
        title='CWT Analysis: Original vs Thresholded with Boundaries',
        height=600,
        width=1200,
    )
    
    # Update axes labels
    fig.update_xaxes(title_text="Gait Cycle (%)", row=1, col=1)
    fig.update_xaxes(title_text="Gait Cycle (%)", row=1, col=2)
    fig.update_yaxes(title_text="Frequency (Hz)", row=1, col=1)
    fig.update_yaxes(title_text="Frequency (Hz)", row=1, col=2)
    
    # Show plot
    st.plotly_chart(fig)
    
    return Z_tr, anotate


# Load the data
record = wfdb.rdrecord("S2")
# Konversi data ke DataFrame
df = pd.DataFrame(record.p_signal, columns=record.sig_name)

st.title("Define Onset Offset from EMG 🐾")

st.sidebar.title("Parameter Control")
# Define sampling parameters
sampling_rate =st.sidebar.number_input("Sampling Rate", value=2000)
sampling_period = 1 / sampling_rate

# Extract a subset of the data
st.sidebar.subheader("Choose One Cycle")
start_index=st.sidebar.number_input("Start Index", value=7042)
end_index  =st.sidebar.number_input("End Index", value=9238)

# Bandpass Filter parameters
st.sidebar.subheader("Bandpass Filter")
cutoff_low = st.sidebar.number_input("Sumbu Vertikal", value=20)    # Lower cutoff frequency in Hz
cutoff_high = st.sidebar.number_input("Sumbu Vertikal", value=450)  # Upper cutoff frequency in Hz
order = st.sidebar.number_input("Sumbu Vertikal", value=2)         # Filter order

st.sidebar.subheader("CWT")
coloumncount=st.sidebar.number_input("Sumbu Vertikal", value=100)
rowcount=st.sidebar.number_input("Sumbu Horizontal", value=100)

# Filter the rows from 7042 to 9238
time = df.index / sampling_rate  
filtered_time = time
filtered_gl = df["semg LT LAT.G"]
filtered_vl = df["semg LT LAT.V"]
filtered_baso_lt_foot = df["baso LT FOOT"]

# Normalize each signal
normalized_gl = normalize_signal(filtered_gl)
normalized_vl = normalize_signal(filtered_vl)
normalized_baso_lt_foot = normalize_signal(filtered_baso_lt_foot)
create_plotly_figure('Original Signal','Time (seconds)','Threshold',time, [normalized_baso_lt_foot], ['Baso Foot'],'lines')

#PLOT GL VL 
create_plotly_figure('GL Original Signal','Time (seconds)','Normalized Amplitude',time, [normalized_gl], ['Gastrocnemius Lateralis (GL)'],'lines')
create_plotly_figure('VL Original Signal','Time (seconds)','Normalized Amplitude',time, [normalized_vl], ['Vastus Lateralis (VL),'],'lines')

t = np.arange(len(df)) / sampling_rate  # Time in seconds
signal_gl = df["semg LT LAT.G"].values  # Gastrocnemius Lateralis (GL)
signal_vl = df["semg LT LAT.V"].values  # Vastus Lateralis (VL)

st.sidebar.subheader("DWT")
level = st.sidebar.number_input("Level", value=8)
threshold = st.sidebar.number_input("Factor DWT", value=20)

st.sidebar.subheader("Thresholding CWT")
threshold_coef_gl=st.sidebar.number_input("Threshold GL", value=0.2)
threshold_coef_vl=st.sidebar.number_input("Threshold VL", value=0.15)


if st.sidebar.button("Start Compute"):
    # Apply bandpass filter to both signals
    df["filtered_gl"] = butterworth_bandpass_filter(signal_gl, cutoff_low, cutoff_high, sampling_period, order)
    df["filtered_vl"] = butterworth_bandpass_filter(signal_vl, cutoff_low, cutoff_high, sampling_period, order)


    # Apply DWT denoising on filtered signals
    df["GL_dwt"] = denoise_dwt(df["filtered_gl"], level, threshold)
    df["VL_dwt"] = denoise_dwt(df["filtered_vl"], level, threshold)

    a = 0.0001
    da = 0.0001
    dt = 1/8000
    f0 = 0.849
    w0 = 2*np.pi* f0

    # Run CWT on denoised signals
    gl_cwt_result = cwt(coloumncount, rowcount, a, da, dt, f0, df["GL_dwt"][start_index:end_index].values)
    vl_cwt_result = cwt(coloumncount, rowcount, a, da, dt, f0, df["VL_dwt"][start_index:end_index].values)


    #PLOT PLOT PLOT PLOT
    st.header("Offset Onset Anotation")
    create_plotly_figure('Original Signal','Time (seconds)','Threshold',time[start_index:end_index], [normalized_baso_lt_foot[start_index:end_index]], ['Baso Foot'],'lines')

    #PLOT GL VL 
    st.header("One Cycle Signal")
    create_plotly_figure('GL Original Signal','Time (seconds)','Normalized Amplitude',time[start_index:end_index], [normalized_gl[start_index:end_index]], ['Gastrocnemius Lateralis (GL)'],'lines')
    create_plotly_figure('VL Original Signal','Time (seconds)','Normalized Amplitude',time[start_index:end_index], [normalized_vl[start_index:end_index]], ['Vastus Lateralis (VL),'],'lines')

    #PLOT BANDPASS 
    st.header("Bandpass Filter")
    create_plotly_figure('GL Bandpass Filter','Time (seconds)','Normalized Amplitude',time[start_index:end_index], [df["semg LT LAT.G"].values[start_index:end_index], df["filtered_gl"][start_index:end_index]], ['Gastrocnemius Lateralis (GL)', 'Filtered Gastrocnemius Lateralis (GL)'],'lines')
    create_plotly_figure('VL Bandpass Filter','Time (seconds)','Normalized Amplitude',time[start_index:end_index], [df["semg LT LAT.V"].values[start_index:end_index], df["filtered_vl"][start_index:end_index]], ['Vastus Lateralis (VL),', 'Filtered Vastus Lateralis (VL),'],'lines')

    #PLOT DWT
    st.header("Discreate Wavelet Transform Filter")
    create_plotly_figure('GL DWT Filter','Time (seconds)','Normalized Amplitude',time[start_index:end_index], [df["filtered_gl"][start_index:end_index], df["GL_dwt"][start_index:end_index]], ['Bandpass Gastrocnemius Lateralis (GL)', 'DWT Gastrocnemius Lateralis (GL)'],'lines')
    create_plotly_figure('VL DWT Filter','Time (seconds)','Normalized Amplitude',time[start_index:end_index], [df["filtered_vl"][start_index:end_index], df["VL_dwt"][start_index:end_index]], ['Bandpass Vastus Lateralis (VL),', 'DWT Vastus Lateralis (VL),'],'lines')

    #PLOT CWT
    st.header("CWT 3D Plot")
    #GL
    X, Y = np.meshgrid(np.arange(rowcount), np.arange(coloumncount))
    Z_gl = gl_cwt_result.T  

    fig_gl = go.Figure(data=[go.Surface(z=Z_gl, x=X, y=Y, colorscale='Viridis')])
    fig_gl.update_layout(title='GL CWT 3D Plot',
                      scene=dict(xaxis_title='Gait Cycle (%)',
                                 yaxis_title='Freq (Hz)',
                                 zaxis_title='Magnitude'),
                      autosize=True)
    st.plotly_chart(fig_gl)

    #VL
    X, Y = np.meshgrid(np.arange(rowcount), np.arange(coloumncount))
    Z_vl = vl_cwt_result.T  

    fig_vl = go.Figure(data=[go.Surface(z=Z_vl, x=X, y=Y, colorscale='Viridis')])
    fig_vl.update_layout(title='VL CWT 3D Plot',
                      scene=dict(xaxis_title='Gait Cycle (%)',
                                 yaxis_title='Freq (Hz)',
                                 zaxis_title='Magnitude'),
                      autosize=True)
    st.plotly_chart(fig_vl)

    # Jalankan fungsi dengan data Z yang sudah ada
    st.header("Thresholding CWT")
    st.subheader("GL Thresholding")
    Z_tr_gl, anotate_gl = plot_thresholded_cwt_with_boundaries(gl_cwt_result.T, threshold_coef_gl)
    st.subheader("VL Thresholding")
    Z_tr_vl, anotate_vl = plot_thresholded_cwt_with_boundaries(vl_cwt_result.T, threshold_coef_vl)

    st.divider()
    st.subheader("Source")
    st.write("Data Availability: [PhysioNet sEMG Data](https://physionet.org/content/semg/1.0.1/)")
    st.write("Paper: [IEEE Acces Paper](https://ieeexplore.ieee.org/document/9673729)")
