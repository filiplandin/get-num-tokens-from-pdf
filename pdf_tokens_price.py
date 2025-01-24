import PyPDF2
import tiktoken # Tiktoken is a fast BPE tokenizer developed by OpenAI,
import argparse

def num_tokens_from_string(string: str) -> int:
    """Returns the number of tokens in a text string."""
    encoding = tiktoken.get_encoding("p50k_base")  # Its for [gpt-4o, gpt-4o-mini, o1-preview, o1-mini, text-embedding-3-large] models
    num_tokens = len(encoding.encode(string))
    return num_tokens

def extract_text_from_pdf(pdf_path: str) -> str:
    """Extracts text from a PDF file."""
    with open(pdf_path, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        text = ''
        for page in reader.pages:
            text += page.extract_text()
    return text

def print_gpt_prices(tokens_count: int, context_length="128K"):
    # Updated pricing for the latest models 17/12/2024
    price_per_1M_tokens = {
        "gpt-4o": 2.50,
        "gpt-4o-mini": 0.150,
        "o1-preview": 15.00,
        "o1-mini": 3.00,
        "text-embedding-3-large": 0.130,
        "TTS": 15.000
    }
    print(f"API costs ({context_length} context and October 2023 knowledge cutoff):")
    for model, price in price_per_1M_tokens.items():
        total_cost = (tokens_count / 1000000) * price
        print(f"{model}: {total_cost:.2f} $")


def main():
    parser = argparse.ArgumentParser(description="Calculate token count and pricing for GPT models from a PDF file.")
    parser.add_argument("pdf_path", type=str, help="Path to the PDF file")
    args = parser.parse_args()

    pdf_path = args.pdf_path
    text_from_pdf = extract_text_from_pdf(pdf_path)
    tokens_count = num_tokens_from_string(text_from_pdf)

    print(f"Number of tokens: {tokens_count}\n")
    print_gpt_prices(tokens_count)

if __name__ == "__main__":
    main()


# 2023 Model pricing information:
# GPT-4: $0.03 per 1,000 tokens
# GPT-3.5 Turbo 4K: $0.0015 per 1,000 tokens
# GPT-3.5 Turbo 16K: $0.003 per 1,000 tokens

#  128K context and an October 2023 knowledge cutoff.
# https://openai.com/api/pricing/