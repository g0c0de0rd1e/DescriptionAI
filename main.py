from transformers import BartForConditionalGeneration, BartTokenizer

# Load model and tokenizer
model_name = "facebook/bart-large-cnn"
model = BartForConditionalGeneration.from_pretrained(model_name)
tokenizer = BartTokenizer.from_pretrained(model_name)

# Short description
short_description = "Прекрасный закат на берегу моря."

# Prepare the text for the model
input_text = f"summarize: {short_description}"
input_ids = tokenizer.encode(input_text, return_tensors="pt")

# Generate the summary
outputs = model.generate(
    input_ids,
    max_length=50,
    min_length=10,
    num_beams=4,
    no_repeat_ngram_size=2,
    length_penalty=1.0,
    early_stopping=True
)

# Decode the generated output
colorful_description = tokenizer.decode(outputs[0], skip_special_tokens=True)

# Debug messages
print("Input text:", input_text)
print("Input IDs:", input_ids)
print("Generated IDs:", outputs)
print("Colorful description:", colorful_description)
