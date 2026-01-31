import pandas as pd
from datasets import load_dataset
import requests
import json
import time
from typing import List, Dict, Any
import os
from tqdm import tqdm

class LMStudioClient:
    """Client for interacting with LM Studio API"""
    
    def __init__(self, base_url: str = "http://localhost:1234", model_name: str = None):
        self.base_url = base_url
        self.model_name = model_name
        self.session = requests.Session()
        
    def generate_paraphrase(self, text: str, max_tokens: int = 4096, temperature: float = 0.7) -> str:
        """Generate a paraphrase for the given text"""
        
        prompt = f"""Simplifique o texto a seguir, mas mantenha o sentido original. Retorne só o texto simplificado.

Texto original: {text}

Texto simplificado: /no_think"""

        payload = {
            "messages": [
                {
                    "role": "system",
                    "content": "Você é um assistente simplificador de textos."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            "temperature": temperature,
            "max_tokens": max_tokens,
            "top_p": 0.8,
            "top_k": 20,
            "stream": False
        }
        
        if self.model_name:
            payload["model"] = self.model_name
            
        try:
            response = self.session.post(
                f"{self.base_url}/v1/chat/completions",
                json=payload,
                timeout=60
            )
            response.raise_for_status()
            
            result = response.json()
            paraphrase = result['choices'][0]['message']['content'].strip()
            
            # Remove "Paraphrase:" prefix if present
            if paraphrase.startswith("<think>\n\n</think>\n"):
                paraphrase = paraphrase[18:].strip()
                
            return paraphrase
            
        except requests.exceptions.RequestException as e:
            print(f"Error generating paraphrase: {e}")
            return None
        except (KeyError, IndexError) as e:
            print(f"Error parsing response: {e}")
            return None

def process_dataset(
    dataset_name: str,
    dataset_config: str,
    text_column: str,
    output_file: str,
    intermediate_file: str = None,
    lm_studio_url: str = "http://localhost:1234",
    model_name: str = None,
    batch_size: int = 100,
    max_samples: int = None,
    delay_between_requests: float = 0.1
):
    """
    Process dataset and generate paraphrases
    
    Args:
        dataset_name: Name of the dataset to load
        dataset_config: Configuration/subset of the dataset
        text_column: Name of the column containing text to paraphrase
        output_file: Path to save the parquet file
        lm_studio_url: Base URL for LM Studio API
        model_name: Specific model name (optional)
        batch_size: Number of samples to process before saving intermediate results
        max_samples: Maximum number of samples to process (None for all)
        delay_between_requests: Delay in seconds between API requests
    """
    
    # Initialize LM Studio client
    client = LMStudioClient(lm_studio_url, model_name)
    
    # Load dataset
    print(f"Loading dataset: {dataset_name}, config: {dataset_config}")
    try:
        ds = load_dataset(dataset_name, dataset_config)
        
        # Use train split if available, otherwise use the first available split
        if 'train' in ds:
            data = ds['train']
        else:
            split_name = list(ds.keys())[0]
            data = ds[split_name]
            print(f"Using split: {split_name}")
            
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return
    
    # Limit samples if specified
    if max_samples and len(data) > max_samples:
        data = data.select(range(247989,247989+max_samples))
        print(f"Limited to {max_samples} samples")
    
    print(f"Processing {len(data)} samples")
    
    # Check if text column exists
    if text_column not in data.column_names:
        print(f"Error: Column '{text_column}' not found in dataset.")
        print(f"Available columns: {data.column_names}")
        return
    
    # Prepare results storage
    results = []
    processed_count = 0
    
    if not intermediate_file:
        # Create progress bar
        pbar = tqdm(total=len(data), desc="Generating paraphrases")

        try:
            for i, sample in enumerate(data, start=247989):
                original_text = sample[text_column]

                # Skip empty texts
                if not original_text or not original_text.strip():
                    pbar.update(1)
                    continue
                
                # Generate paraphrase
                paraphrase = client.generate_paraphrase(original_text)

                if paraphrase:
                    results.append({
                        'original_text': original_text,
                        'paraphrase': paraphrase,
                        'sample_id': i
                    })
                    processed_count += 1
                else:
                    print(f"Failed to generate paraphrase for sample {i}")

                # Save intermediate results
                if len(results) % batch_size == 0 and results:
                    save_intermediate_results(results, output_file, processed_count)

                # Add delay to avoid overwhelming the API
                time.sleep(delay_between_requests)

                pbar.update(1)

        except KeyboardInterrupt:
            print("\nInterrupted by user. Saving current progress...")

        finally:
            pbar.close()

            # Save final results
            if results:
                save_final_results(results, output_file)
                print(f"\nProcessing complete! Generated {len(results)} paraphrases.")
                print(f"Results saved to: {output_file}")
            else:
                print("No paraphrases were generated.")
                
    else:
        df = pd.read_parquet(intermediate_file)
        id_samples = df["sample_id"].tolist()
        # Create progress bar
        pbar = tqdm(total=len(data), desc="Generating paraphrases")

        try:
            for i, sample in enumerate(data):
                if i in id_samples:
                    pbar.update(1)
                    continue
                    
                original_text = sample[text_column]

                # Skip empty texts
                if not original_text or not original_text.strip():
                    pbar.update(1)
                    continue
                
                # Generate paraphrase
                paraphrase = client.generate_paraphrase(original_text)

                if paraphrase:
                    results.append({
                        'original_text': original_text,
                        'paraphrase': paraphrase,
                        'sample_id': i
                    })
                    processed_count += 1
                else:
                    print(f"Failed to generate paraphrase for sample {i}")

                # Save intermediate results
                if len(results) % batch_size == 0 and results:
                    save_intermediate_results(results, output_file, processed_count)

                # Add delay to avoid overwhelming the API
                time.sleep(delay_between_requests)

                pbar.update(1)

        except KeyboardInterrupt:
            print("\nInterrupted by user. Saving current progress...")

        finally:
            pbar.close()

            # Save final results
            if results:
                save_final_combined_results(results, output_file, df)
                print(f"\nProcessing complete! Generated {len(results)} paraphrases.")
                print(f"Results gathered and saved to: {output_file}")
            else:
                print("No paraphrases were generated.")

def save_intermediate_results(results: List[Dict], output_file: str, processed_count: int):
    """Save intermediate results to avoid losing progress"""
    intermediate_file = output_file.replace('.parquet', f'_intermediate_{processed_count}.parquet')
    df = pd.DataFrame(results)
    df.to_parquet(intermediate_file, index=False)
    print(f"\nSaved intermediate results: {len(results)} pairs to {intermediate_file}")

def save_final_combined_results(results: List[Dict], output_file: str, df_original):
    """Save combined results to parquet file"""
    df = pd.DataFrame(results)
    df_combined = pd.concat([df_original, df], ignore_index=True).sort_values(by="sample_id")
    df_combined.to_parquet(output_file, index=False)
    
    # Clean up intermediate files
    directory = os.path.dirname(output_file) or '.'
    base_name = os.path.basename(output_file).replace('.parquet', '')
    
    for file in os.listdir(directory):
        if file.startswith(f"{base_name}_intermediate_") and file.endswith('.parquet'):
            os.remove(os.path.join(directory, file))
            print(f"Cleaned up intermediate file: {file}")


def save_final_results(results: List[Dict], output_file: str):
    """Save final results to parquet file"""
    df = pd.DataFrame(results)
    df.to_parquet(output_file, index=False)
    
    # Clean up intermediate files
    directory = os.path.dirname(output_file) or '.'
    base_name = os.path.basename(output_file).replace('.parquet', '')
    
    for file in os.listdir(directory):
        if file.startswith(f"{base_name}_intermediate_") and file.endswith('.parquet'):
            os.remove(os.path.join(directory, file))
            print(f"Cleaned up intermediate file: {file}")

def main():
    """Main function with example usage"""
    
    # Configuration
    config = {
        'dataset_name': "eduagarcia/LegalPT_dedup",
        'dataset_config': "acordaos_tcu",
        'text_column': "text",  # Adjust this based on your dataset structure
        'output_file': "paraphrased_dataset.parquet",
        'lm_studio_url': "http://localhost:1234",
        'model_name': None,  # Set specific model if needed
        'batch_size': 50,
        'max_samples': None,  # Set to a number to limit processing for testing
        'delay_between_requests': 0.2  # Adjust based on your API limits
    }
    
    # Start processing
    process_dataset(**config)

if __name__ == "__main__":
    # Example: Process a small subset for testing
    test_config = {
        'dataset_name': "eduagarcia/LegalPT_dedup",
        'dataset_config': "tesemo_v2",
        'text_column': "text",  # You may need to adjust this
        'output_file': "tesemo_v2_v4.parquet",
        'intermediate_file': "",
        'lm_studio_url': "http://localhost:1234",
        'model_name': None,
        'batch_size': 2000,
        'max_samples': 126331,  # Process only 50 samples for testing
        'delay_between_requests': 0.1
    }
    
    print("Starting paraphrase generation...")
    print("Make sure LM Studio is running with a model loaded!")
    print(f"Testing with {test_config['max_samples']} samples")
    
    process_dataset(**test_config)
