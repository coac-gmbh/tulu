import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

def main():
    # === Konfiguration ===
    model_dir = "meta-llama/Llama-3.2-1B-Instruct"
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # === Modell und Tokenizer laden ===
    tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_dir,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32
    ).to(device)
    model.eval()

    print("✅ Modell geladen. Gib deinen Prompt ein (zum Beenden: Ctrl+C):")

    while True:
        try:
            user_input = input("\n🧠 Prompt: ")
            if not user_input.strip():
                continue

            # Optional: Chat-Template verwenden (falls gewünscht)
            messages = [{"role": "user", "content": user_input}]
            prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

            inputs = tokenizer(prompt, return_tensors="pt").to(device)

            with torch.no_grad():
                output = model.generate(
                    **inputs,
                    max_new_tokens=200,
                    do_sample=False,
                    temperature=0.7
                )

            generated = tokenizer.decode(output[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True)
            print(f"\n🤖 Antwort:\n{generated.strip()}")

        except KeyboardInterrupt:
            print("\n❎ Beendet.")
            break

if __name__ == "__main__":
    main()
