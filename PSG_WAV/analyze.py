from osa_data_only_phone import OSAData

data_path = r"C:\V89\data"
processor = OSAData(data_path)
processor.load_data()

print(f"Loaded {len(processor.patient_audio)} patients")

# ดูข้อมูลผู้ป่วยคนแรก
wav_data_1, sr_1 = processor.patient_audio[0]
print(wav_data_1.shape)       # (จำนวน samples,)
print(sr_1)                   # เช่น 48000
# Now you can use processor.patient_audio, processor.patient_annotations, etc.