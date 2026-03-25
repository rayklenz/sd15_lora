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
    # Фиксируем seed для воспроизводимости
    seed = cfg.get("seed", 42)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # 1. Загружаем базовую модель
    print(f"📥 Загрузка модели {cfg.model.pretrained_model_name}...")
    pipe = StableDiffusionPipeline.from_pretrained(
        cfg.model.pretrained_model_name,
        torch_dtype=torch.float32,
        safety_checker=None
    )

    # 2. Загружаем LoRA
    lora_path = cfg.inferencer.ckpt_dir
    print(f"📥 Загрузка LoRA из {lora_path}...")
    if os.path.exists(os.path.join(lora_path, "adapter_model.safetensors")):
        pipe.unet = PeftModel.from_pretrained(pipe.unet, lora_path)
        print("✅ LoRA загружена!")
    else:
        print(f"⚠️ Файлы LoRA не найдены в {lora_path}")

    pipe.to(cfg.trainer.device)
    pipe.enable_attention_slicing()

    # =====================================================
    # 🎨 НАСТРОЙКИ (те же, что и в исходном коде)
    # =====================================================
    GUIDANCE_SCALE = 8.0
    LORA_STRENGTH = 1.0
    NUM_INFERENCE_STEPS = cfg.inferencer.num_inference_steps
    RETRY_CROPPED = True
    MAX_RETRIES = 3

    # =====================================================
    # 👁️ ЛИЦО — ОБЯЗАТЕЛЬНО
    # =====================================================
    FACE_BOOST = "(face visible:1.4), (bird face:1.3), (expressive face:1.3), (detailed face:1.3), (eyes visible:1.4), (beak visible:1.4), (facial features:1.3)"
    ANATOMY_BOOST = "(perfect bird anatomy:1.3), (proper proportions:1.3), (natural pose:1.3), (realistic structure:1.3), (two legs:1.4), (two wings:1.4), (single beak:1.4), (long tail:1.3), (small curved beak:1.3), (blue or purple cere:1.3), (black and yellow striped head:1.3), (cheek patches:1.3)"
    COMPOSITION_FULL = f"bird fully in frame, entire bird visible, whole body shown, centered, good composition, well-framed, no cropping, {FACE_BOOST}"
    COMPOSITION_PORTRAIT = f"close-up, face visible, head fully in frame, beak visible, portrait composition, centered face, {FACE_BOOST}"
    COMPOSITION_MEDIUM = f"medium shot, upper body visible, well-framed, bird centered, head fully visible, {FACE_BOOST}"
    UNIVERSAL_POSITIVE = "sharp focus, high quality, detailed, beautiful"

    # НЕГАТИВНЫЕ ПРОМПТЫ (усилены)
    ANATOMY_NEGATIVE = "bad anatomy, deformed, distorted, unnatural pose, extra limbs, wrong proportions, malformed, three legs, four legs, extra wings, missing legs, two beaks, extra beak, no beak"
    COMPOSITION_NEGATIVE = "cropped, out of frame, partially visible, incomplete, truncated, cut off, missing parts, head cut off, body cut off, edge of frame, frame edge, head out of frame, beak cut off"
    FACE_NEGATIVE = "no face, face hidden, face obscured, head hidden, no eyes, face not visible"
    QUALITY_NEGATIVE = "blurry, low quality, ugly, distorted, pixelated, artifacts"
    STYLE_NEGATIVE = "oversaturated, neon, rainbow, gaudy, spiky, thorny, cactus-like, rough texture, heavy, solid"

    NEGATIVE_BOOST = f"{ANATOMY_NEGATIVE}, {COMPOSITION_NEGATIVE}, {FACE_NEGATIVE}, {QUALITY_NEGATIVE}, {STYLE_NEGATIVE}, (no face:1.8), (face hidden:1.8), (blurred face:1.8), (deformed face:1.8), (caterpillar:1.8), (slug:1.8), (merged bodies:1.8), (conjoined:1.8), (fused:1.8), (extra legs:1.8), (missing legs:1.8), (no eyes:1.8)"

    # =====================================================
    # 🎭 НОВЫЕ СТИЛИ ДЛЯ ВОЛНИСТЫХ ПОПУГАЙЧИКОВ
    # =====================================================
    styles = [
        # 🎀 МИЛЫЕ И ПАСТЕЛЬНЫЕ
        {
            "name": "Нежная лаванда",
            "prompt": f"a cute sks bird, budgerigar, soft lavender and cream colors, pastel, dreamy atmosphere, delicate, kawaii style, fluffy feathers, {ANATOMY_BOOST}, {COMPOSITION_FULL}, {UNIVERSAL_POSITIVE}",
            "negative": NEGATIVE_BOOST + ", dark, aggressive, sharp, realistic, scary"
        },
        {
            "name": "Сахарная вата",
            "prompt": f"a sks bird, budgerigar, cotton candy colors, soft pink and mint green, fluffy texture, sweet, magical, fairy tale style, {ANATOMY_BOOST}, {COMPOSITION_FULL}, {UNIVERSAL_POSITIVE}",
            "negative": NEGATIVE_BOOST + ", dark, scary, realistic, aggressive"
        },
        {
            "name": "Лунная нежность",
            "prompt": f"a sks bird, budgerigar, moonlight pastel, silver and soft blue, gentle glow, ethereal, dreamy, peaceful, {ANATOMY_BOOST}, {COMPOSITION_FULL}, {UNIVERSAL_POSITIVE}",
            "negative": NEGATIVE_BOOST + ", dark, bright, aggressive, scary"
        },
        {
            "name": "Весенняя зелень",
            "prompt": f"a cute sks bird, budgerigar, soft lime and yellow, spring pastels, fresh, gentle, adorable, {ANATOMY_BOOST}, {COMPOSITION_FULL}, {UNIVERSAL_POSITIVE}",
            "negative": NEGATIVE_BOOST + ", dark, aggressive, scary"
        },
        # 🌈 ВЕСЁЛЫЕ И ЯРКИЕ
        {
            "name": "Радужный попугай",
            "prompt": f"a joyful sks bird, budgerigar, rainbow colors, vibrant, happy, celebration, colorful feathers, cheerful, {ANATOMY_BOOST}, {COMPOSITION_FULL}, {UNIVERSAL_POSITIVE}",
            "negative": NEGATIVE_BOOST + ", dark, sad, dull, gloomy, scary"
        },
        {
            "name": "Солнечный лучик",
            "prompt": f"a happy sks bird, budgerigar, bright yellow and gold, sunshine, energetic, playful, summer vibe, radiant, {ANATOMY_BOOST}, {COMPOSITION_FULL}, {UNIVERSAL_POSITIVE}",
            "negative": NEGATIVE_BOOST + ", dark, sad, gloomy, night, scary"
        },
        {
            "name": "Фестиваль красок",
            "prompt": f"a joyful sks bird, budgerigar, neon colors, party vibe, energetic, celebration, vibrant, explosive colors, fun, {ANATOMY_BOOST}, {COMPOSITION_FULL}, {UNIVERSAL_POSITIVE}",
            "negative": NEGATIVE_BOOST + ", dark, sad, dull, scary, aggressive"
        },
        {
            "name": "Австралийское солнце",
            "prompt": f"a happy sks bird, budgerigar, bright green and yellow, australian outback, sunny, cheerful, wild budgie, {ANATOMY_BOOST}, {COMPOSITION_FULL}, {UNIVERSAL_POSITIVE}",
            "negative": NEGATIVE_BOOST + ", dark, sad, gloomy, scary"
        },
        # 🎈 ИГРИВЫЕ И БЫСТРЫЕ (здесь используем композицию "medium shot", чтобы подчеркнуть движение)
        {
            "name": "Вихрь перьев",
            "prompt": f"a playful sks bird, budgerigar, dynamic motion, blurred wings, speed lines, energetic, action, fun, mischievous, flying fast, {ANATOMY_BOOST}, {COMPOSITION_MEDIUM}, {UNIVERSAL_POSITIVE}",
            "negative": NEGATIVE_BOOST + ", static, still, sad, dark, scary"
        },
        {
            "name": "Озорной волнистик",
            "prompt": f"a mischievous sks bird, budgerigar, swirling colors, playful, dynamic pose, cartoon style, energetic, fun, {ANATOMY_BOOST}, {COMPOSITION_MEDIUM}, {UNIVERSAL_POSITIVE}",
            "negative": NEGATIVE_BOOST + ", still, sad, dark, aggressive, scary"
        },
        {
            "name": "Летняя беззаботность",
            "prompt": f"a playful sks bird, budgerigar, bright turquoise and lime, summer vibes, carefree, energetic, joyful, dynamic, {ANATOMY_BOOST}, {COMPOSITION_MEDIUM}, {UNIVERSAL_POSITIVE}",
            "negative": NEGATIVE_BOOST + ", dark, sad, gloomy, aggressive, scary"
        },
        {
            "name": "Быстрый полет",
            "prompt": f"a swift sks bird, budgerigar, speed motion, dynamic blur, energetic, action shot, fast flying, {ANATOMY_BOOST}, {COMPOSITION_MEDIUM}, {UNIVERSAL_POSITIVE}",
            "negative": NEGATIVE_BOOST + ", static, still, slow, sad"
        },
        # 🌑 ЗЛЫЕ И ТЁМНЫЕ
        {
            "name": "Грозовая туча",
            "prompt": f"an angry sks bird, budgerigar, dark storm clouds, lightning, dramatic lighting, menacing, powerful, intense colors, {ANATOMY_BOOST}, {COMPOSITION_FULL}, {UNIVERSAL_POSITIVE}",
            "negative": NEGATIVE_BOOST + ", bright, happy, cute, pastel, peaceful"
        },
        {
            "name": "Теневой волнистик",
            "prompt": f"a fierce sks bird, budgerigar, dark and ominous, black and deep blue, glowing red eyes, aggressive, threatening, gothic style, {ANATOMY_BOOST}, {COMPOSITION_FULL}, {UNIVERSAL_POSITIVE}",
            "negative": NEGATIVE_BOOST + ", cute, bright, happy, pastel, peaceful"
        },
        {
            "name": "Пламя гнева",
            "prompt": f"an enraged sks bird, budgerigar, fiery colors, red and black, intense, dramatic, angry, powerful, aggressive stance, {ANATOMY_BOOST}, {COMPOSITION_FULL}, {UNIVERSAL_POSITIVE}",
            "negative": NEGATIVE_BOOST + ", cute, happy, pastel, bright, peaceful"
        },
        {
            "name": "Ночной охотник",
            "prompt": f"a dark sks bird, budgerigar, shadows, gothic style, mysterious, haunting, deep purple and black, eerie, intense, {ANATOMY_BOOST}, {COMPOSITION_FULL}, {UNIVERSAL_POSITIVE}",
            "negative": NEGATIVE_BOOST + ", cute, bright, happy, pastel, peaceful"
        },
        # 😌 СПОКОЙНЫЕ И МЕДИТАТИВНЫЕ
        {
            "name": "Утренняя роса",
            "prompt": f"a calm sks bird, budgerigar, soft greens and blues, zen, peaceful, meditative, nature, quiet morning, serene, {ANATOMY_BOOST}, {COMPOSITION_FULL}, {UNIVERSAL_POSITIVE}",
            "negative": NEGATIVE_BOOST + ", aggressive, dark, scary, bright, loud"
        },
        {
            "name": "Лесное спокойствие",
            "prompt": f"a peaceful sks bird, budgerigar, forest colors, moss green and earth tones, tranquil, harmony, nature vibe, {ANATOMY_BOOST}, {COMPOSITION_FULL}, {UNIVERSAL_POSITIVE}",
            "negative": NEGATIVE_BOOST + ", aggressive, scary, bright, neon, loud"
        },
        {
            "name": "Мечтательный волнистик",
            "prompt": f"a dreamy sks bird, budgerigar, soft watercolor, gentle blues, relaxed, peaceful, thoughtful, {ANATOMY_BOOST}, {COMPOSITION_FULL}, {UNIVERSAL_POSITIVE}",
            "negative": NEGATIVE_BOOST + ", aggressive, scary, loud, bright"
        },
        # 💕 РОМАНТИЧНЫЕ И НЕЖНЫЕ
        {
            "name": "Валентинов день",
            "prompt": f"two separate sks bird, budgerigars, each with distinct body, romantic scene, heart shapes, red and pink colors, love, valentine style, tender, soft lighting, kissing, two heads, four legs, two beaks, all faces visible, {ANATOMY_BOOST}, {COMPOSITION_FULL}, {UNIVERSAL_POSITIVE}",
            "negative": NEGATIVE_BOOST + ", aggressive, scary, dark, sad, alone, merged, conjoined, fused, one body with two heads, extra legs, missing legs, no face, blurred face, caterpillar-like, slug-like, no eyes, deformed"
        },
        {
            "name": "Облако влюбленных",
            "prompt": f"two separate sks bird, budgerigars, each with distinct body, together, soft pink clouds, romantic, couple, adorable, affectionate, lovebirds style, cuddling, two heads, four legs, two beaks, all faces visible, {ANATOMY_BOOST}, {COMPOSITION_FULL}, {UNIVERSAL_POSITIVE}",
            "negative": NEGATIVE_BOOST + ", alone, aggressive, scary, dark, merged, conjoined, fused, one body with two heads, extra legs, missing legs, no face, blurred face, caterpillar-like, slug-like, no eyes, deformed"
        },
        {
            "name": "Нежное прикосновение",
            "prompt": f"a loving sks bird, budgerigar, soft peach and cream colors, romantic, gentle, affectionate, tender moment, sweet budgie, {ANATOMY_BOOST}, {COMPOSITION_FULL}, {UNIVERSAL_POSITIVE}",
            "negative": NEGATIVE_BOOST + ", aggressive, dark, scary, lonely"
        },
        # ✨ МАГИЧЕСКИЕ И ЗАГАДОЧНЫЕ
        {
            "name": "Секретная магия",
            "prompt": f"a mysterious sks bird, budgerigar, magical aura, purple and gold, mystical, enigmatic, secret power, enchanting, fantasy budgie, {ANATOMY_BOOST}, {COMPOSITION_FULL}, {UNIVERSAL_POSITIVE}",
            "negative": NEGATIVE_BOOST + ", ordinary, ugly, blurry, low quality"
        },
        {
            "name": "Звездный странник",
            "prompt": f"a magical sks bird, budgerigar, starry night, cosmic, celestial, galaxy wings, mystical, dreamy, fantasy, space budgie, {ANATOMY_BOOST}, {COMPOSITION_FULL}, {UNIVERSAL_POSITIVE}",
            "negative": NEGATIVE_BOOST + ", ordinary, ugly, blurry, low quality"
        },
        {
            "name": "Волшебный волнистик",
            "prompt": f"an enchanted sks bird, budgerigar, magic sparkles, fairy dust, mystical glow, fantasy art, magical creature, spellbinding, {ANATOMY_BOOST}, {COMPOSITION_FULL}, {UNIVERSAL_POSITIVE}",
            "negative": NEGATIVE_BOOST + ", ordinary, ugly, blurry, realistic"
        },
        # 🤪 ЭКСЦЕНТРИЧНЫЕ И БЕЗУМНЫЕ
        {
            "name": "Хаотичная радость",
            "prompt": f"a crazy sks bird, budgerigar, chaotic colors, splashes, unpredictable, fun, psychedelic, wild, energetic mess, silly budgie, {ANATOMY_BOOST}, {COMPOSITION_FULL}, {UNIVERSAL_POSITIVE}",
            "negative": NEGATIVE_BOOST + ", calm, organized, simple, boring"
        },
        {
            "name": "Карнавал безумия",
            "prompt": f"a wild sks bird, budgerigar, carnival style, mismatched patterns, bright chaos, fun, crazy, unique, eccentric, party budgie, {ANATOMY_BOOST}, {COMPOSITION_FULL}, {UNIVERSAL_POSITIVE}",
            "negative": NEGATIVE_BOOST + ", calm, ordinary, simple, boring"
        },
        {
            "name": "Сумасшедший волнистик",
            "prompt": f"a crazy bird, budgerigar, surreal colors, absurd poses, funny, whimsical, cartoon chaos, hilarious, budgie antics, {ANATOMY_BOOST}, {COMPOSITION_FULL}, {UNIVERSAL_POSITIVE}",
            "negative": NEGATIVE_BOOST + ", calm, normal, realistic, boring"
        },
        # 🎨 СПЕЦИАЛЬНЫЕ ДЛЯ ВОЛНИСТЫХ ПОПУГАЕВ (портретные, используем COMPOSITION_PORTRAIT)
        {
            "name": "Классический волнистый",
            "prompt": f"a beautiful sks bird, budgerigar, classic green and yellow, black scalloped markings, blue cere, detailed feathers, realistic, {ANATOMY_BOOST}, {COMPOSITION_PORTRAIT}, {UNIVERSAL_POSITIVE}",
            "negative": NEGATIVE_BOOST + ", ugly, blurry, low quality, cartoon"
        },
        {
            "name": "Голубой волнистик",
            "prompt": f"a stunning sks bird, budgerigar, sky blue feathers, white face, black markings, cobalt, beautiful, detailed, {ANATOMY_BOOST}, {COMPOSITION_PORTRAIT}, {UNIVERSAL_POSITIVE}",
            "negative": NEGATIVE_BOOST + ", ugly, blurry, low quality"
        },
        {
            "name": "Лютино (желтый)",
            "prompt": f"a beautiful sks bird, budgerigar, lutino variety, bright yellow, red eyes, sunshine, vibrant, stunning, rare color, {ANATOMY_BOOST}, {COMPOSITION_PORTRAIT}, {UNIVERSAL_POSITIVE}",
            "negative": NEGATIVE_BOOST + ", ugly, blurry, low quality, green"
        },
        {
            "name": "Альбинос",
            "prompt": f"a beautiful sks bird, budgerigar, albino variety, pure white, red eyes, pristine, elegant, rare, beautiful, angelic, {ANATOMY_BOOST}, {COMPOSITION_PORTRAIT}, {UNIVERSAL_POSITIVE}",
            "negative": NEGATIVE_BOOST + ", ugly, blurry, low quality, color"
        },
        {
            "name": "Опалин",
            "prompt": f"a beautiful sks bird, budgerigar, opaline variety, soft gradient colors, pastel, elegant, rare mutation, stunning feathers, {ANATOMY_BOOST}, {COMPOSITION_PORTRAIT}, {UNIVERSAL_POSITIVE}",
            "negative": NEGATIVE_BOOST + ", ugly, blurry, low quality"
        },
        {
            "name": "Хохлатый волнистик",
            "prompt": f"a cute sks bird, budgerigar, crested variety, fluffy head feathers, adorable, rare mutation, cute, charming, unique, {ANATOMY_BOOST}, {COMPOSITION_PORTRAIT}, {UNIVERSAL_POSITIVE}",
            "negative": NEGATIVE_BOOST + ", ugly, blurry, low quality"
        }
    ]

    # Перемешивание стилей
    random.shuffle(styles)
    print("🎲 Стили перемешаны в случайном порядке!")

    os.makedirs("generated_images_budgie", exist_ok=True)

    print("\n" + "=" * 60)
    print(f"🎨 ГЕНЕРАЦИЯ ВОЛНИСТЫХ ПОПУГАЙЧИКОВ (новые стили)")
    print(f"   ✅ LoRA сила: {LORA_STRENGTH}")
    print(f"   ✅ Guidance scale: {GUIDANCE_SCALE}")
    print(f"   👁️ ЛИЦО: ОБЯЗАТЕЛЬНО во всех кадрах!")
    print(f"   🛡️ Защита от обрезания: ВКЛЮЧЕНА")
    print(f"   🎲 Всего стилей: {len(styles)}")
    print("=" * 60)

    # =====================================================
    # 🔍 ФУНКЦИЯ ПРОВЕРКИ ОБРЕЗАНИЯ (та же)
    # =====================================================
    def check_if_cropped(image, threshold=0.05):
        img_array = np.array(image)
        h, w = img_array.shape[:2]
        # Проверяем края
        edge_top = img_array[0:10, :].mean()
        edge_bottom = img_array[-10:, :].mean()
        edge_left = img_array[:, 0:10].mean()
        edge_right = img_array[:, -10:].mean()
        center = img_array[h // 2 - 20:h // 2 + 20, w // 2 - 20:w // 2 + 20].mean()
        edge_avg = (edge_top + edge_bottom + edge_left + edge_right) / 4
        diff = abs(edge_avg - center)
        # Дополнительно: если верхний край яркий (значит там не фон), то считаем обрезанным
        top_bright = edge_top > (center * 0.8)  # порог подберите
        return (diff < 10) or top_bright

    # =====================================================
    # 🎨 ГЕНЕРАЦИЯ
    # =====================================================
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
                    height=768,
                    width=768,
                    cross_attention_kwargs={"scale": LORA_STRENGTH}
                ).images[0]

            if image.mode != "RGB":
                image = image.convert("RGB")

            is_cropped = check_if_cropped(image)
            if not is_cropped:
                best_image = image
                best_cropped = False
                break
            else:
                if best_image is None:
                    best_image = image
                print(f"   ⚠️ Попытка {attempt+1}: обрезано, перегенерируем...")

        if best_cropped:
            cropped_warning = True
            print(f"   ⚠️ После {MAX_RETRIES+1} попыток изображение всё ещё может быть обрезано.")
        else:
            print(f"   ✅ Изображение в кадре.")

        images.append(best_image)
        filename = f"generated_images_budgie/{style['name'].replace(' ', '_')}.png"
        best_image.save(filename)
        print(f"   💾 Сохранено: {filename}")

    # =====================================================
    # 🖼️ ВИЗУАЛИЗАЦИЯ
    # =====================================================
    print("\n🖼️ СОЗДАЮ КОЛЛАЖ...")
    n = len(styles)
    rows = (n + 5) // 6
    cols = min(6, n)
    fig, axes = plt.subplots(rows, cols, figsize=(18, 3*rows))
    if rows*cols > 1:
        axes = axes.flatten()
    else:
        axes = [axes]

    for i, (style, img) in enumerate(zip(styles, images)):
        axes[i].imshow(img)
        axes[i].set_title(f"{style['name']}", fontsize=9)
        axes[i].axis('off')

    for i in range(len(images), rows*cols):
        axes[i].axis('off')

    plt.tight_layout()
    plt.savefig("generated_images_budgie/all_styles_collage.png", dpi=150, bbox_inches='tight')
    plt.show()

    print("\n" + "=" * 60)
    print("✅ ГОТОВО! Результаты в папке generated_images_budgie/")
    print(f"📸 Всего сгенерировано: {len(images)} изображений")
    if cropped_warning:
        print("\n⚠️ Некоторые изображения могут быть обрезаны (несмотря на попытки перегенерации).")
    else:
        print("\n✅ Все изображения выглядят корректно!")

if __name__ == "__main__":
    main()
