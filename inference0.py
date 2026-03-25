import hydra
from omegaconf import DictConfig
from diffusers import StableDiffusionPipeline
from peft import PeftModel
import torch
from PIL import Image
import os
import matplotlib.pyplot as plt
import random
import numpy as np

@hydra.main(version_base=None, config_path="src/configs", config_name="persongen_inference_lora_b")
def main(cfg: DictConfig):
    seed = cfg.get("seed", 42)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    print(f"📥 Загрузка модели {cfg.model.pretrained_model_name}...")
    pipe = StableDiffusionPipeline.from_pretrained(
        cfg.model.pretrained_model_name,
        torch_dtype=torch.float32,
        safety_checker=None
    )

    lora_path = cfg.inferencer.ckpt_dir
    print(f"📥 Загрузка LoRA из {lora_path}...")
    if os.path.exists(os.path.join(lora_path, "adapter_model.safetensors")):
        pipe.unet = PeftModel.from_pretrained(pipe.unet, lora_path)
        print("✅ LoRA загружена!")
    else:
        print(f"⚠️ Файлы LoRA не найдены в {lora_path}")

    pipe.to(cfg.trainer.device)
    pipe.enable_attention_slicing()

    # ========== НАСТРОЙКИ ==========
    GUIDANCE_SCALE = 7.5
    LORA_STRENGTH = 0.7
    NUM_INFERENCE_STEPS = cfg.inferencer.num_inference_steps
    MAX_RETRIES = 3

    # ========== КОРОТКИЕ БУСТЫ ==========
    ANATOMY_BOOST = "detailed face, sharp beak, clear eyes"
    UNIVERSAL_POSITIVE = "high resolution, 8k, masterpiece"

    # Негативы
    NEGATIVE_BOOST = "blurry, deformed, mutated, low quality, bad anatomy, extra limbs, worst quality, cropped, out of frame"

    # ========== СТИЛИ БЕЗ КОМПОЗИЦИОННЫХ ВСТАВОК ==========
    styles = [
        # 🎀 МИЛЫЕ И ПАСТЕЛЬНЫЕ
        {"name": "Нежная лаванда", "prompt": f"single sks budgierigar, soft lavender pastel feathers, dreamy, kawaii, fluffy, {ANATOMY_BOOST}, {UNIVERSAL_POSITIVE}", "negative": NEGATIVE_BOOST + ", dark, aggressive, realistic"},
        {"name": "Сахарная вата", "prompt": f"single sks budgierigar, cotton candy pink and mint green, sweet, fluffy texture, {ANATOMY_BOOST}, {UNIVERSAL_POSITIVE}", "negative": NEGATIVE_BOOST + ", dark, scary, realistic"},
        {"name": "Лунная нежность", "prompt": f"single sks budgierigar, ethereal, silver and soft blue glowing feathers, centered, {ANATOMY_BOOST}, {UNIVERSAL_POSITIVE}", "negative": NEGATIVE_BOOST + ", dark, bright, aggressive"},
        {"name": "Весенняя зелень", "prompt": f"single sks bird, soft lime and yellow, spring pastels, fresh, gentle, {ANATOMY_BOOST}, {UNIVERSAL_POSITIVE}", "negative": NEGATIVE_BOOST + ", dark, aggressive"},

        # 🌈 ВЕСЁЛЫЕ И ЯРКИЕ
        {"name": "Радужный попугай", "prompt": f"single sks budgierigar, rainbow vibrant colors, colorful feathers, cheerful, {ANATOMY_BOOST}, {UNIVERSAL_POSITIVE}", "negative": NEGATIVE_BOOST + ", dark, sad, gloomy"},
        {"name": "Солнечный лучик", "prompt": f"single sks budgierigar, bright yellow and gold, shining feathers, centered, {ANATOMY_BOOST}, {UNIVERSAL_POSITIVE}", "negative": NEGATIVE_BOOST + ", dark, sad, gloomy"},
        {"name": "Фестиваль красок", "prompt": f"single sks budgierigar, bright colorful feathers, vibrant, centered, {ANATOMY_BOOST}, {UNIVERSAL_POSITIVE}", "negative": NEGATIVE_BOOST + ", dark, sad, dull"},
        {"name": "Австралийское солнце", "prompt": f"single sks budgierigar, bright green and yellow feathers, sunny, wild, {ANATOMY_BOOST}, {UNIVERSAL_POSITIVE}", "negative": NEGATIVE_BOOST + ", dark, sad, gloomy"},

        # 🎈 ИГРИВЫЕ И БЫСТРЫЕ (движение, но без композиции)
        {"name": "Вихрь перьев", "prompt": f"single sks budgierigar, flying, wings spread, dynamic motion, in flight, centered, {ANATOMY_BOOST}, {UNIVERSAL_POSITIVE}", "negative": NEGATIVE_BOOST + ", static, still, sad"},
        {"name": "Озорной волнистик", "prompt": f"single sks budgierigar, dynamic pose, energetic, fun, {ANATOMY_BOOST}, {UNIVERSAL_POSITIVE}", "negative": NEGATIVE_BOOST + ", still, sad, dark"},
        {"name": "Летняя беззаботность", "prompt": f"single sks budgierigar, bright turquoise and lime feathers, centered, clear eyes, {ANATOMY_BOOST}, {UNIVERSAL_POSITIVE}", "negative": NEGATIVE_BOOST + ", dark, sad, gloomy"},
        {"name": "Быстрый полет", "prompt": f"single sks budgierigar, flying fast, dynamic, {ANATOMY_BOOST}, {UNIVERSAL_POSITIVE}", "negative": NEGATIVE_BOOST + ", static, still, slow"},

        # 🌑 ЗЛЫЕ И ТЁМНЫЕ
        {"name": "Грозовая туча", "prompt": f"single sks budgierigar, clouds, dark mood, intense expression, {ANATOMY_BOOST}, {UNIVERSAL_POSITIVE}", "negative": NEGATIVE_BOOST + ", bright, happy, cute, pastel"},
        {"name": "Теневой волнистик", "prompt": f"single sks budgierigar, black and deep blue feathers, glowing red eyes, {ANATOMY_BOOST}, {UNIVERSAL_POSITIVE}", "negative": NEGATIVE_BOOST + ", cute, bright, happy, pastel"},
        {"name": "Пламя гнева", "prompt": f"single sks budgierigar, fiery red and orange feathers, smoke, centered, {ANATOMY_BOOST}, {UNIVERSAL_POSITIVE}", "negative": NEGATIVE_BOOST + ", cute, happy, pastel, bright"},
        {"name": "Ночной охотник", "prompt": f"single sks budgierigar, shadows, mysterious, deep purple and black feathers, {ANATOMY_BOOST}, {UNIVERSAL_POSITIVE}", "negative": NEGATIVE_BOOST + ", cute, bright, happy, pastel"},

        # 😌 СПОКОЙНЫЕ И МЕДИТАТИВНЫЕ
        {"name": "Утренняя роса", "prompt": f"single sks budgierigar, soft greens and blues, zen, peaceful, nature, centered, {ANATOMY_BOOST}, {UNIVERSAL_POSITIVE}", "negative": NEGATIVE_BOOST + ", aggressive, dark, bright, loud"},
        {"name": "Лесное спокойствие", "prompt": f"single sks budgierigar, forest colors, moss green and earth tones, tranquil, {ANATOMY_BOOST}, {UNIVERSAL_POSITIVE}", "negative": NEGATIVE_BOOST + ", aggressive, dark, bright, loud"},
        {"name": "Мечтательный волнистик", "prompt": f"single sks budgierigar, watercolor style, gentle blues, relaxed, centered, {ANATOMY_BOOST}, {UNIVERSAL_POSITIVE}", "negative": NEGATIVE_BOOST + ", aggressive, dark, bright, loud"},

        # 💕 РОМАНТИЧНЫЕ (две птицы)
        {"name": "Валентинов день", "prompt": f"two sks budgierigar, romantic scene, heart shapes, red and pink, soft lighting, kissing, {ANATOMY_BOOST}, {UNIVERSAL_POSITIVE}", "negative": NEGATIVE_BOOST + ", aggressive, dark, sad, alone, merged, conjoined"},
        {"name": "Облако влюбленных", "prompt": f"two sks budgierigar, soft pink clouds, romantic, couple, cuddling, {ANATOMY_BOOST}, {UNIVERSAL_POSITIVE}", "negative": NEGATIVE_BOOST + ", alone, aggressive, dark, merged, conjoined"},
        {"name": "Нежное прикосновение", "prompt": f"single sks budgierigar, soft peach and cream colors, romantic, gentle, affectionate, centered, {ANATOMY_BOOST}, {UNIVERSAL_POSITIVE}", "negative": NEGATIVE_BOOST + ", aggressive, dark, lonely"},

        # ✨ МАГИЧЕСКИЕ
        {"name": "Секретная магия", "prompt": f"single sks budgierigar, glowing feathers, purple and gold, mystical, centered, {ANATOMY_BOOST}, {UNIVERSAL_POSITIVE}", "negative": NEGATIVE_BOOST + ", ordinary, ugly, blurry"},
        {"name": "Звездный странник", "prompt": f"single sks budgierigar, starry night sky, cosmic, galaxy wings, ethereal, {ANATOMY_BOOST}, {UNIVERSAL_POSITIVE}", "negative": NEGATIVE_BOOST + ", ordinary, ugly, blurry"},
        {"name": "Волшебный волнистик", "prompt": f"single sks budgierigar, magic sparkles, fairy dust, glowing feathers, {ANATOMY_BOOST}, {UNIVERSAL_POSITIVE}", "negative": NEGATIVE_BOOST + ", ordinary, ugly, blurry"},

        # 🤪 ЭКСЦЕНТРИЧНЫЕ
        {"name": "Хаотичная радость", "prompt": f"single sks budgierigar, chaotic colors, splashes, wild, fun, {ANATOMY_BOOST}, {UNIVERSAL_POSITIVE}", "negative": NEGATIVE_BOOST + ", calm, organized, simple"},
        {"name": "Карнавал безумия", "prompt": f"single sks budgierigar, vibrant, colorful, festive, fun, eccentric, centered, {ANATOMY_BOOST}, {UNIVERSAL_POSITIVE}", "negative": NEGATIVE_BOOST + ", calm, ordinary, simple"},
        {"name": "Сумасшедший волнистик", "prompt": f"single sks budgierigar, whimsical, playful, funny, cartoon style, bright colors, centered, {ANATOMY_BOOST}, {UNIVERSAL_POSITIVE}", "negative": NEGATIVE_BOOST + ", calm, normal, realistic"},

        # 🎨 СПЕЦ-ОКРАСЫ (портретные, но без макросов)
        {"name": "Классический волнистый", "prompt": f"single sks budgierigar, green and yellow, blue cere,  {ANATOMY_BOOST}, {UNIVERSAL_POSITIVE}", "negative": NEGATIVE_BOOST + ", ugly, blurry, cartoon"},
        {"name": "Голубой волнистик", "prompt": f"single sks budgierigar, sky blue feathers, white face, black markings, {ANATOMY_BOOST}, {UNIVERSAL_POSITIVE}", "negative": NEGATIVE_BOOST + ", ugly, blurry"},
        {"name": "Лютино (желтый)", "prompt": f"single sks budgierigar, lutino variety, solid bright yellow, red eyes, {ANATOMY_BOOST}, {UNIVERSAL_POSITIVE}", "negative": NEGATIVE_BOOST + ", ugly, blurry, green"},
        {"name": "Альбинос", "prompt": f"single sks budgierigar, albino variety, pure white, red eyes, elegant, {ANATOMY_BOOST}, {UNIVERSAL_POSITIVE}", "negative": NEGATIVE_BOOST + ", ugly, blurry, color"},
        {"name": "Опалин", "prompt": f"single sks budgierigar, opaline variety, soft gradient colors, pastel, {ANATOMY_BOOST}, {UNIVERSAL_POSITIVE}", "negative": NEGATIVE_BOOST + ", ugly, blurry"},
        {"name": "Хохлатый волнистик", "prompt": f"single sks budgierigar, crested variety, fluffy head feathers, adorable, {ANATOMY_BOOST}, {UNIVERSAL_POSITIVE}", "negative": NEGATIVE_BOOST + ", ugly, blurry"}
    ]

    random.shuffle(styles)
    os.makedirs("generated_images_budgie_final_nocomp", exist_ok=True)

    print("\n" + "="*60)
    print(f"🎨 ФИНАЛЬНАЯ ГЕНЕРАЦИЯ (без композиционных вставок)")
    print(f"   ✅ LoRA сила: {LORA_STRENGTH}")
    print(f"   ✅ Guidance scale: {GUIDANCE_SCALE}")
    print(f"   ✅ Шагов: {NUM_INFERENCE_STEPS}")
    print(f"   📸 Всего стилей: {len(styles)}")
    print("="*60)

    def check_if_cropped(image):
        img_array = np.array(image)
        h, w = img_array.shape[:2]
        edge_top = img_array[0:10, :].mean()
        edge_bottom = img_array[-10:, :].mean()
        edge_left = img_array[:, 0:10].mean()
        edge_right = img_array[:, -10:].mean()
        center = img_array[h//2-20:h//2+20, w//2-20:w//2+20].mean()
        edge_avg = (edge_top + edge_bottom + edge_left + edge_right) / 4
        diff = abs(edge_avg - center)
        top_bright = edge_top > (center * 0.8)
        return (diff < 10) or top_bright

    images = []
    cropped_warning = False

    for i, style in enumerate(styles):
        print(f"\n🎨 {i+1}/{len(styles)}: {style['name']}")
        print(f"   Промпт: {style['prompt'][:80]}...")

        best_image = None
        best_cropped = True
        for attempt in range(MAX_RETRIES+1):
            with torch.no_grad():
                image = pipe(
                    style["prompt"],
                    num_inference_steps=NUM_INFERENCE_STEPS,
                    guidance_scale=GUIDANCE_SCALE,
                    negative_prompt=style["negative"],
                    height=512,
                    width=512,
                    cross_attention_kwargs={"scale": LORA_STRENGTH}
                ).images[0]

            if image.mode != "RGB":
                image = image.convert("RGB")

            if not check_if_cropped(image):
                best_image = image
                best_cropped = False
                break
            else:
                if best_image is None:
                    best_image = image
                print(f"   ⚠️ Попытка {attempt+1}: обрезано, перегенерируем...")

        if best_cropped:
            cropped_warning = True
            print(f"   ⚠️ После {MAX_RETRIES+1} попыток изображение может быть обрезано.")
        else:
            print(f"   ✅ Изображение в кадре.")

        images.append(best_image)
        filename = f"generated_images_budgie_final_nocomp/{style['name'].replace(' ', '_')}.png"
        best_image.save(filename)
        print(f"   💾 Сохранено: {filename}")

    print("\n🖼️ СОЗДАЮ КОЛЛАЖ...")
    n = len(styles)
    rows = (n + 5) // 6
    cols = min(6, n)
    fig, axes = plt.subplots(rows, cols, figsize=(18, 3*rows))
    axes = axes.flatten() if rows*cols > 1 else [axes]

    for i, (style, img) in enumerate(zip(styles, images)):
        axes[i].imshow(img)
        axes[i].set_title(f"{style['name']}", fontsize=9)
        axes[i].axis('off')

    for i in range(len(images), rows*cols):
        axes[i].axis('off')

    plt.tight_layout()
    plt.savefig("generated_images_budgie_final_nocomp/all_styles_collage.png", dpi=150, bbox_inches='tight')
    plt.show()

    print("\n" + "="*60)
    print("✅ ГОТОВО! Результаты в папке generated_images_budgie_final_nocomp/")
    print(f"📸 Всего сгенерировано: {len(images)} изображений")

if __name__ == "__main__":
    main()
