import streamlit as st
from transformers import WhisperProcessor, WhisperForConditionalGeneration
import torch
from pydub import AudioSegment
import librosa
import os

# تابع تبدیل خودکار به فرمت WAV
def convert_to_wav(input_audio_path):
    file_name, file_extension = os.path.splitext(input_audio_path)
    if file_extension.lower() != ".wav":
        output_audio_path = file_name + ".wav"
        audio = AudioSegment.from_file(input_audio_path)
        audio.export(output_audio_path, format="wav")
        return output_audio_path
    else:
        return input_audio_path

# تنظیمات مدل و دستگاه پردازش
model_path = "C:/Users/FZL/Downloads/whisper"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# بارگذاری مدل Whisper و پردازنده
model = WhisperForConditionalGeneration.from_pretrained(model_path).to(device)
processor = WhisperProcessor.from_pretrained(model_path)

# رابط کاربری Streamlit
st.set_page_config(page_title="تبدیل صوت به متن", page_icon=":sound:", layout="wide")

# اعمال CSS برای وسط‌چین کردن محتوا و راست‌چین کردن متن
st.markdown("""
    <style>
    /* تنظیمات برای وسط چین کردن محتوای صفحه */
    .main {
        display: flex;
        justify-content: center;
        align-items: center;
        padding-top: 50px;
        text-align: center;
    }

    /* تنظیمات راست‌چین برای متن‌ها */
    .rtl-text {
        direction: rtl;
        text-align: right;
        font-size: 18px;
        font-family: "IRANSans", sans-serif;
    }

    </style>
    """, unsafe_allow_html=True)

# هدر
st.image("https://your-image-url.com/banner.png", use_column_width=True)
st.title("🔊 تبدیل صوت به متن با Whisper")

st.markdown("""
    این ابزار به شما امکان می‌دهد که فایل‌های صوتی خود را به متن تبدیل کنید. 
    کافیست یک فایل صوتی با فرمت **WAV**, **MP3**, **OGG** یا **FLAC** آپلود کنید و خروجی متن را دریافت کنید.
    """)

# آپلود فایل صوتی
st.markdown("### 1️⃣ آپلود فایل صوتی")
audio_file = st.file_uploader("فایل صوتی خود را انتخاب کنید:", type=["wav", "mp3", "ogg", "flac"])

if audio_file is not None:
    # ذخیره فایل آپلود شده به صورت موقت
    with open("temp_audio_file", "wb") as f:
        f.write(audio_file.getbuffer())

    # نشانگر پیشرفت
    with st.spinner('در حال پردازش فایل صوتی...'):
        # تبدیل فایل به فرمت WAV (در صورت نیاز)
        wav_audio_file = convert_to_wav("temp_audio_file")

        # خواندن فایل صوتی با librosa و استخراج داده‌های عددی
        audio, sampling_rate = librosa.load(wav_audio_file, sr=16000)

        # پردازش داده‌های صوتی برای مدل Whisper
        audio_input = processor(audio, sampling_rate=sampling_rate, return_tensors="pt").input_features.to(device)

        # تولید متن از صوت
        predicted_ids = model.generate(audio_input)
        transcribed_text = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]

        # حذف فایل موقت پس از پردازش
        os.remove("temp_audio_file")

    # نمایش متن تولید شده در حالت راست‌چین
    st.markdown("### 2️⃣ متن تولید شده:")
    st.markdown(f"<div class='rtl-text'>{transcribed_text}</div>", unsafe_allow_html=True)

    # دکمه دانلود متن
    st.markdown("### 3️⃣ دانلود متن:")
    st.download_button(label="📥 دانلود متن", data=transcribed_text, file_name="transcription.txt", mime="text/plain")

else:
    st.warning("لطفاً یک فایل صوتی آپلود کنید.")
