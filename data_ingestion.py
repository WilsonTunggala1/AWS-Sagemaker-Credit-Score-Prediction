import os
import pandas as pd
from pathlib import Path

class DataIngestion:
    def __init__(self, input_path: str | Path, output_dir: str | Path):
        self.input_file = Path(input_path)
        self.output_dir = Path(output_dir)
        self.output_file = self.output_dir / "data_A.csv"

    def run(self) -> Path:
        print("Data Ingestion")
        
        self.output_dir.mkdir(parents=True, exist_ok=True)

        try:
            df = pd.read_csv(self.input_file)
        except FileNotFoundError:
            raise FileNotFoundError(f"❌ File sumber tidak ditemukan di path: {self.input_file}")

        assert not df.empty, "❌ Dataset kosong. Periksa kembali isi file data_A.csv Anda."
        print(f"Dataset berhasil dimuat dengan dimensi: {df.shape}")

        df.to_csv(self.output_file, index=False)
        print(f"✅ Data berhasil di-ingest dari {self.input_file} → {self.output_file}\n")
        
        return self.output_file


if __name__ == "__main__":
    container_input = "/opt/ml/processing/input"
    container_output = "/opt/ml/processing/ingested"
    
    if os.path.exists(container_input):
        input_dir = Path(container_input)
        output_dir = Path(container_output)
    else:
        input_dir = Path("/home/ec2-user/SageMaker/uas")
        output_dir = Path("/home/ec2-user/SageMaker/uas/ingested")
    
    input_file = input_dir / "data_A.csv"
    
    print(f"Looking for file at: {input_file}")
    
    if input_file.exists():
        ingestor = DataIngestion(input_path=input_file, output_dir=output_dir)
        ingestor.run()
    else:
        print(f"❌ Error: {input_file} not found!")

        if input_dir.exists():
            print(f"Environment AWS saat ini melihat file berikut di {input_dir}: {os.listdir(input_dir)}")
        else:
            print(f"Direktori {input_dir} belum ada/belum di-mount.")