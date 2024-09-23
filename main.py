from transformers import GPT2LMHeadModel, GPT2Tokenizer
import time
import warnings
import re

warnings.filterwarnings("ignore", category=FutureWarning)

def generate_marketing_text(product_description):
    model_name = "sberbank-ai/rugpt3small_based_on_gpt2"
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    model = GPT2LMHeadModel.from_pretrained(model_name)

    template = f"""
    Продукт: {product_description}
    Целевая аудитория: потенциальные покупатели
    Стиль: убедительный, информативный
    Тон: убедительный, вдохновляющий

    Напишите маркетинговое описание для {product_description}. Акцентируйтесь на его ключевых преимуществах и способности удовлетворить потребности целевой аудитории. Используйте конкретные детали и примеры, демонстрирующие, как {product_description} может улучшить жизнь или бизнес пользователя.

    Описание: """

    inputs = tokenizer.encode(template, return_tensors="pt")

    max_generation_length = 250
    output = inputs
    generated_text = ""

    start_time = time.time()

    outputs = model.generate(
        input_ids=output,
        max_new_tokens=max_generation_length,
        do_sample=True,  
        temperature=0.6,  
        top_p=0.95,  
        repetition_penalty=1.05,  # штраф за повторения
        num_return_sequences=1,
        return_dict_in_generate=True,
        output_scores=True,
    )

    generated_text = tokenizer.decode(outputs.sequences[0], skip_special_tokens=True)
    generated_text = re.sub(r'[^\w\s]', ' ', generated_text)

    sentences = re.split(r'(?<=[.!?]) +', generated_text)
    unique_sentences = []

    for sentence in sentences:
        if sentence.strip() not in unique_sentences:
            unique_sentences.append(sentence.strip())

    completed_text = ' '.join(unique_sentences)

    def complete_sentences(text):
        sentences = re.split(r'(?<=[.!?]) +', text)
        completed_sentences = []
        
        for sentence in sentences:
            if not sentence.strip().endswith(('.','?','!')):
                completed_sentence = sentence.strip() + '.'
            else:
                completed_sentence = sentence.strip()
            
            completed_sentences.append(completed_sentence)
        
        return ' '.join(completed_sentences)

    completed_text = complete_sentences(generated_text)

    final_text = completed_text

    end_time = time.time()
    generation_time = end_time - start_time

    print(f"\nВремя генерации: {generation_time:.2f} секунд")
    return final_text

# Пример использования
product_description = input("Введите краткое описание вашего продукта или услуги: ")
result = generate_marketing_text(product_description)
print("\nРезультат:")
print(result)
