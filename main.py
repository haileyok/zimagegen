from typing import Any, Dict, List, Tuple
import torch
from diffusers import ZImagePipeline
import random
import json
from pathlib import Path
from tqdm import tqdm
import time

OUTPUT_DIR = Path("./generated_images")
METADATA_FILE = OUTPUT_DIR / "metadata.jsonl"
BATCH_SIZE = 4
IMAGE_SIZE = 256
NUM_INFERENCE_STEPS = 9

PROMPT_TEMPLATES: Dict[str, List[str]] = {
    "portraits": [
        "a portrait photograph of {subject}",
        "headshot of {subject}, professional lighting",
        "close-up portrait of {subject}, studio lighting",
        "{subject}, portrait photography, 50mm lens",
        "candid portrait of {subject}, natural lighting",
    ],
    "landscapes": [
        "a photograph of {scene}",
        "{scene}, landscape photography, golden hour",
        "scenic view of {scene}, wide angle",
        "{scene}, nature photography, high detail",
        "panoramic view of {scene}, sunset lighting",
    ],
    "objects": [
        "product photography of {object}",
        "a photograph of {object} on white background",
        "{object}, studio photography, professional",
        "close-up of {object}, macro photography",
        "{object}, commercial photography, high quality",
    ],
    "animals": [
        "a photograph of {animal}",
        "{animal} in natural habitat, wildlife photography",
        "close-up of {animal}, nature documentary style",
        "{animal}, professional wildlife photography",
        "portrait of {animal}, detailed fur/feathers",
    ],
    "architecture": [
        "architectural photography of {building}",
        "{building}, exterior view, architectural digest style",
        "interior of {building}, wide angle photography",
        "{building}, modern architecture photography",
        "detail shot of {building}, architectural photography",
    ],
    "food": [
        "food photography of {dish}",
        "{dish}, overhead view, professional food styling",
        "close-up of {dish}, gourmet presentation",
        "{dish}, restaurant quality, appetizing",
        "artful presentation of {dish}, culinary photography",
    ],
    "abstract": [
        "abstract art with {concept}",
        "{concept}, abstract expressionism",
        "geometric abstract art featuring {concept}",
        "colorful abstract composition with {concept}",
        "minimalist abstract art, {concept}",
    ],
    "scenes": [
        "a photograph of {location}",
        "{location}, documentary photography",
        "street photography in {location}",
        "{location}, photojournalism style",
        "everyday life in {location}, candid photography",
    ],
}

PROMPT_SUBJECTS: Dict[str, List[str]] = {
    "portraits": [
        "a young woman",
        "an elderly man",
        "a child",
        "a teenager",
        "a business professional",
        "an artist",
        "a scientist",
        "an athlete",
        "a person wearing glasses",
        "someone smiling",
        "a serious expression",
        "a person with curly hair",
        "someone in traditional clothing",
    ],
    "landscapes": [
        "mountains at sunset",
        "a forest in autumn",
        "a beach at sunrise",
        "desert sand dunes",
        "a waterfall",
        "snowy peaks",
        "rolling hills",
        "a lake surrounded by trees",
        "dramatic clouds over plains",
        "coastal cliffs",
        "a valley",
        "misty mountains",
    ],
    "objects": [
        "a coffee cup",
        "a vintage camera",
        "a smartphone",
        "a book",
        "a watch",
        "headphones",
        "a plant",
        "a chair",
        "a lamp",
        "sunglasses",
        "a keyboard",
        "a pen",
        "shoes",
        "a backpack",
    ],
    "animals": [
        "a cat",
        "a dog",
        "a bird",
        "a lion",
        "an elephant",
        "a fox",
        "a bear",
        "a deer",
        "a tiger",
        "a wolf",
        "an owl",
        "a horse",
        "a rabbit",
        "a squirrel",
        "a butterfly",
    ],
    "architecture": [
        "a modern skyscraper",
        "a historic cathedral",
        "a cozy cottage",
        "a minimalist house",
        "an art deco building",
        "a glass office building",
        "a brick warehouse",
        "a wooden cabin",
        "a concrete structure",
        "a contemporary museum",
        "a traditional temple",
        "a stone castle",
    ],
    "food": [
        "a gourmet burger",
        "sushi platter",
        "pasta carbonara",
        "chocolate cake",
        "fresh salad",
        "grilled steak",
        "pizza",
        "ramen bowl",
        "fruit tart",
        "breakfast plate",
        "tacos",
        "ice cream sundae",
        "seafood platter",
        "vegetable curry",
    ],
    "abstract": [
        "swirling colors",
        "geometric shapes",
        "flowing lines",
        "vibrant gradients",
        "textured patterns",
        "bold contrasts",
        "organic forms",
        "crystalline structures",
        "fluid dynamics",
        "fractal patterns",
        "color blocks",
        "wave patterns",
    ],
    "scenes": [
        "a busy city street",
        "a quiet cafe",
        "a park on a sunny day",
        "a train station",
        "a farmers market",
        "an office workspace",
        "a library",
        "a concert venue",
        "a shopping mall",
        "a classroom",
        "a hospital corridor",
        "a subway platform",
    ],
}

STYLE_MODIFIERS: List[str] = [
    "",
    ", dramatic lighting",
    ", soft lighting",
    ", black and white",
    ", vibrant colors",
    ", muted tones",
    ", high contrast",
    ", shallow depth of field",
    ", sharp focus",
    ", cinematic",
    ", photorealistic",
    ", 8k resolution",
    ", professional photography",
]


def generate_random_prompt() -> Tuple[str, str]:
    category = random.choice(list(PROMPT_TEMPLATES.keys()))
    template = random.choice(PROMPT_TEMPLATES[category])
    subject = random.choice(PROMPT_SUBJECTS[category])

    prompt = template.format(
        subject=subject,
        scene=subject,
        object=subject,
        animal=subject,
        building=subject,
        dish=subject,
        concept=subject,
        location=subject,
    )

    if random.random() < 0.5:
        modifier = random.choice(STYLE_MODIFIERS)
        prompt += modifier

    return prompt, category


def generate_params() -> Dict[str, Any]:
    return {
        "num_inference_steps": random.choice([7, 8, 9, 10]),
        "height": random.choice([768, 512, 256]),
        "width": random.choice([768, 512, 256]),
        "seed": random.randint(0, 2**32 - 1),
    }


def generate_dataset(num_images: int):
    print("Loading Z-Image pipeline...")
    pipe = ZImagePipeline.from_pretrained(
        "Tongyi-MAI/Z-Image-Turbo",
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=False,
    )
    pipe.to("cuda")

    print(f"Starting output of {num_images} images...")
    print(f"Output directory: {OUTPUT_DIR}")

    generation_times: List[float] = []
    category_counts: Dict[str, int] = {}

    with open(METADATA_FILE, "w") as meta_file:
        for i in tqdm(range(num_images), desc="Generating images"):
            try:
                prompt, category = generate_random_prompt()
                params = generate_params()

                category_counts[category] = category_counts.get(category, 0) + 1

                start_time = time.time()

                image = pipe(
                    prompt=prompt,
                    height=params["height"],
                    width=params["width"],
                    num_inference_steps=params["num_inference_steps"],
                    guidance_scale=0.0,
                    generator=torch.Generator("cuda").manual_seed(params["seed"]),
                ).images[0]

                generation_time = time.time() - start_time
                generation_times.append(generation_time)

                image_filename = f"image_{i:06d}.png"
                image_path = OUTPUT_DIR / image_filename
                image.save(image_path)

                metadata = {
                    "image_id": i,
                    "filename": image_filename,
                    "prompt": prompt,
                    "category": category,
                    "seed": params["seed"],
                    "height": params["height"],
                    "width": params["width"],
                    "num_inference_steps": params["num_inference_steps"],
                    "generation_time": generation_time,
                }
                meta_file.write(json.dumps(metadata) + "\n")
                meta_file.flush()

                if (i + 1) % 10 == 0:
                    avg_time = sum(generation_times[-10:]) / len(generation_times[-10:])
                    remaining = num_images - (i + 1)
                    eta_seconds = remaining * avg_time
                    eta_hours = eta_seconds / 3600

                    print("\n--- Progress Update ---")
                    print(f"Generated: {i + 1}/{num_images}")
                    print(f"Avg time per image: {avg_time:.2f}s")
                    print(f"ETA: {eta_hours:.2f} hours")
                    print(f"Category distribution: {category_counts}")

            except Exception as e:
                print(f"\nError generating image {i}: {e}")
                continue

    print("\n" + "=" * 50)
    print("GENERATION COMPLETE!")
    print("=" * 50)
    print(f"Total images: {num_images}")
    print(
        f"Average generation time: {sum(generation_times) / len(generation_times):.2f}s"
    )
    print(f"Total time: {sum(generation_times) / 3600:.2f} hours")
    print("\nCategory distribution:")
    for category, count in sorted(category_counts.items()):
        print(f"  {category}: {count} ({100 * count / num_images:.1f}%)")
    print(f"\nImages saved to: {OUTPUT_DIR}")
    print(f"Metadata saved to: {METADATA_FILE}")


def count_existing_images():
    """Count how many images already exist"""
    if not OUTPUT_DIR.exists():
        return 0
    return len(list(OUTPUT_DIR.glob("image_*.png")))


def resume_generation():
    """Resume generation if interrupted"""
    existing = count_existing_images()
    if existing > 0:
        response = input(
            f"Found {existing} existing images. Resume from there? (y/n): "
        )
        if response.lower() == "y":
            return existing
    return 0


def main():
    start_idx = resume_generation()

    OUTPUT_DIR.mkdir(exist_ok=True)
    METADATA_FILE.touch(exist_ok=True)

    num_images = 50_000

    if start_idx > 0:
        print(f"Resuming from image {start_idx}")
        num_images = num_images - start_idx

    generate_dataset(num_images)


if __name__ == "__main__":
    main()
