import os
import time
import pandas as pd
from tqdm import tqdm
from typing import List
from openai import OpenAI
from dotenv import load_dotenv
from prompts import INSTRUCTIONS_PROMPT
from data_templates import MountainBatch
from generation_utils import calc_total_costs

load_dotenv()


class Generator:
    def __init__(self, api_client=OpenAI()):
        self.openai_client = api_client


    def __llm_api_call(self, prompt:str, model:str = "gpt-5.4-mini"):
        response = self.openai_client.beta.chat.completions.parse(
            model=model,
            messages=[
                {"role": "system", "content": "You are a helpful synthetic data generation assistant."},
                {"role": "user", "content": prompt}
            ],
            response_format=MountainBatch
        )
        return response.choices[0].message.parsed, response.usage


    def __correct_markers(self, text: str, entity_names: List[str], markers: List[List[int]]) -> List[List[int]]:
        """
        Programmatically self-corrects marker indices by anchoring around the LLM's
        suggested spans, safeguarding against overlapping spans and substring mismatches.
        """
        corrected_markers = []
        for i, name in enumerate(entity_names):
            llm_suggested_start = markers[i][0] if i < len(markers) else 0

            # Locate all occurrences of this name in the text
            occurrences = []
            start = 0
            while True:
                idx = text.find(name, start)
                if idx == -1:
                    break
                occurrences.append(idx)
                start = idx + 1

            if occurrences:
                # Anchor to the occurrence closest to the LLM's recommended index
                best_start = min(occurrences, key=lambda x: abs(x - llm_suggested_start))
                # Append a list instead of a tuple
                corrected_markers.append([best_start, best_start + len(name)])
            else:
                # Fallback to the original span if name is completely missing
                if i < len(markers):
                    corrected_markers.append(markers[i])

        return corrected_markers


    def generate_synthetic_mountains(self, total_records=1000, batch_size=50):
        all_records = list()
        all_usages = list()
        num_batches = total_records // batch_size

        print(f"Starting generation of {total_records} records in {num_batches} batches...")

        for batch_idx in tqdm(range(num_batches)):
            try:

                instructions_prompt = INSTRUCTIONS_PROMPT.format(batch_size=batch_size)
                batch_data, usage = self.__llm_api_call(prompt=instructions_prompt)
                all_usages.append(usage)

                # Apply marker correction
                for record in batch_data.records:
                    markers = self.__correct_markers(record.text, record.entity_names, record.markers)

                    all_records.append({
                        "text": record.text,
                        "marker": markers
                    })

                # Respect rate limits between batches
                time.sleep(1.0)

            except Exception as e:
                print(f"\nError generating batch {batch_idx + 1}: {e}")
                continue

        df_synthetic = pd.DataFrame(all_records)
        return df_synthetic, all_usages



if __name__ == "__main__":
    generator = Generator()
    df_synthetic, usages = generator.generate_synthetic_mountains()

    os.makedirs(os.environ["DATA_DIR"], exist_ok=True)
    df_synthetic.to_csv(os.environ["DATA_PATH"], index=False)
    print(f"\nSuccessfully generated and saved {len(df_synthetic)} valid records!")

    total_costs = calc_total_costs(usages)
    print(f"\nTotal costs: {total_costs:.2f} $$$")

    print("\nExample data:")
    print(df_synthetic.head(3))