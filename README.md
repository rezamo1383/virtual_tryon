# Multi-Domain AI Generation Platform

یک Backend ماژولار و Production-oriented برای محصولات تولید و ویرایش تصویر.
محصول فعلی Virtual Clothing Try-On بدون بازنویسی پشت معماری tenant-aware قرار
گرفته و Wallpaper Visualization به‌عنوان pipeline قابل توسعه اضافه شده است.
FastAPI، CLI، Settings، logging، validation، retry، مدیریت Job و فایل‌های موقت
زیرساخت مشترک همهٔ دامنه‌ها هستند.

## معماری

```text
FastAPI / CLI / shared infrastructure
                  ↓
Tenant Resolver (API key → TenantConfig)
                  ↓
Task Router (tenant.pipeline)
                  ↓
ClothingPipeline | WallpaperPipeline
                  ↓
Domain Validator + Prompt Builder
                  ↓
Model Router (tenant provider/model overrides)
                  ↓
Analysis / Generation Provider
                  ↓
Shared Job, Output, Retry and Logging infrastructure
```

API هیچ `task_type` دریافت نمی‌کند. هویت tenant فقط از `X-API-Key` یا
`Authorization: Bearer ...` به‌دست می‌آید و `TenantConfig.pipeline` دامنه را
تعیین می‌کند. `TaskRouter` از FastAPI مستقل است و pipelineهای tenant را به‌صورت
lazy singleton می‌سازد.

نام‌های داخلی عمومی‌اند:

```text
source_image + reference_image

clothing: source → person, reference → garment
wallpaper: source → room, reference → wallpaper
```

`ClothingPipeline` یک adapter کوچک روی `VirtualTryOnPipeline` موجود است؛ موتور
لباس، retry، evaluation، preprocessing و ساختار خروجی قبلی حفظ شده‌اند.
`WallpaperPipeline` قراردادهای مجزای wall analysis، wall segmentation،
perspective estimation، texture repetition، generation و lighting preservation
دارد. backend پیش‌فرض `semantic` با SegFormer/ADE20K و به‌صورت محلی یک mask
پیکسلی از تمام دیوارهای قابل‌مشاهده می‌سازد؛ در نتیجه پنجره، در، تلویزیون،
کابینت و مبلمان از mask خارج می‌مانند و این مرحله توکن مصرف نمی‌کند. حالت
قدیمی و کم‌دقت‌تر `polygon` همچنان قابل انتخاب است. Generation از همان provider
routing، candidate، retry و مدیریت خروجی مشترک استفاده می‌کند.

امتیاز نهایی در برنامه و مستقل از `overall_score` مدل محاسبه می‌شود:

```text
identity 30% + garment 25% + color 20% + body 15% + background 10%

wall coverage 20% + pattern 25% + perspective 20% + lighting 15% + scene 20%
```

## ساختار پوشه‌ها

```text
virtual_tryon/
├── main.py, cli.py, api.py, config.py
├── config/
│   └── tenants.example.json
├── app/
│   ├── core/          # settings, runtime, logging, exceptions
│   ├── tenant/        # TenantConfig, store, API-key resolver
│   ├── routing/       # TaskRouter, PromptRouter, ModelRouter
│   ├── pipelines/     # BasePipeline, ClothingPipeline, WallpaperPipeline
│   ├── prompts/       # domain-owned prompt builders
│   ├── providers/     # tenant-scoped provider factory
│   ├── validators/    # common, clothing, wallpaper validators
│   ├── models/        # shared/domain Pydantic contracts
│   ├── clients/       # provider protocols, adapters and mocks
│   ├── preprocessing/ # CPU-first local vision and normalization
│   ├── services/      # reusable processing, scoring and legacy engine
│   └── utils/         # safe files, JSON, image and hashing
├── inputs/{persons,garments,rooms,wallpapers}/
├── outputs/
├── models/         # ignored local model cache
├── temp/
├── logs/
└── tests/
```

Job لباس ساختار زیر را می‌سازد:

```text
outputs/{job_id}/
├── request.json
├── preprocessing/
│   ├── preprocessing.json
│   ├── person/{normalized,transparent,foreground_mask,replace_mask,...}.png
│   └── garment/{normalized,transparent_cropped,garment_mask}.png
├── garment_analysis.json
├── garment_mask.png
├── variants/{color}.png
├── candidates/{color}/candidate_01.png
├── final/{color}.png
├── candidate_metadata.json
└── results.json
```

Job کاغذ دیواری:

```text
outputs/{job_id}/
├── request.json
├── wall_analysis.json
├── wallpaper/
│   ├── wall_mask.png
│   ├── wall_debug.png
│   └── texture_perspective.png
├── candidates/attempt_00/candidate_01.png
├── final/wallpaper.png
├── candidate_metadata.json
└── results.json
```

## نصب

Python 3.11 یا جدیدتر لازم است.

```bash
python -m venv .venv
```

Windows:

```powershell
.venv\Scripts\activate
```

Linux/macOS:

```bash
source .venv/bin/activate
```

سپس:

```bash
pip install -r requirements.txt
```

برای preprocessing محلی کامل، dependencyهای CPU را نصب کنید:

```bash
python -m pip install -r requirements-preprocessing.txt
```

این فایل MediaPipe Pose سازگار با Windows x86-64/Python 3.11، rembg و
ONNX Runtime CPU را نصب می‌کند. پروژه مستقیماً از
`opencv-python-headless` استفاده می‌کند و وابستگی اجباری به CUDA یا PyTorch
ندارد. MediaPipe 0.10.21 به‌دلیل داشتن wheel رسمی CPython 3.11 برای Windows
x86-64 pin شده است.

اگر environment قدیمی شامل `opencv-python` GUI باشد، آن را با محیط مجازی تازه
جدا کنید؛ نصب هم‌زمان چند distribution با namespace مشترک `cv2` قابل اتکا نیست.
محیط فعلی سیستم نیز dependencyهای نامرتبط و ناسازگار دارد، بنابراین venv تازه
توصیه می‌شود.

## تنظیم `.env`

```bash
copy .env.example .env
```

در Linux:

```bash
cp .env.example .env
```

مسیرها نسبت به working directory تفسیر می‌شوند. محدودیت حجم، ابعاد، نسبت تصویر،
تعداد Candidate، threshold پذیرش و timeoutها همگی در `.env` قابل تنظیم‌اند. کلیدها
فقط باید در environment یا `.env` محلی قرار گیرند؛ `.env` در Git نادیده گرفته شده
است.

تحلیل شخص در مسیر عادی غیرفعال است؛ بنابراین هیچ درخواست Vision-LLM برای شخص
ارسال و هیچ `person_analysis.json` تولید نمی‌شود. در `results.json` نیز مقدار
`person_analysis` برابر `null` است:

```env
PERSON_ANALYSIS_ENABLED=false
```

برای فعال‌سازی مجدد تحلیل و ساخت `person_analysis.json`، مقدار بالا را `true`
کنید. در آن حالت `REJECT_UNSUITABLE_PERSON_IMAGES` مشخص می‌کند تصویر نامناسب
متوقف شود یا به تولید ادامه دهد. تغییر تنظیمات نیازمند راه‌اندازی مجدد برنامه است.

## Tenant Configuration

تنظیمات پایه برای سازگاری با نسخهٔ قبلی:

```env
TENANT_CONFIG_PATH=config/tenants.json
DEFAULT_TENANT_ID=legacy-clothing
TENANT_AUTH_REQUIRED=false
```

اگر فایل tenant وجود نداشته باشد، runtime دو tenant داخلی
`legacy-clothing=clothing` و `wallpaper-demo=wallpaper` می‌سازد. برای production:

```powershell
Copy-Item "config\tenants.example.json" "config\tenants.json"
python -c "import getpass,hashlib; print(hashlib.sha256(getpass.getpass('API key: ').encode()).hexdigest())"
```

خروجی SHA-256 را در `api_key_sha256` tenant قرار دهید و سپس:

```env
TENANT_AUTH_REQUIRED=true
```

فایل واقعی `config/tenants.json` در Git نادیده گرفته می‌شود. API key خام نه در
فایل tenant ذخیره می‌شود و نه log می‌شود. هر tenant دقیقاً یک مقدار `pipeline`
دارد و می‌تواند `analysis_provider`، `generation_provider`،
`analysis_model`، `generation_model`، `prompt_profile` و `feature_flags` مستقل
داشته باشد.

## Local Image Preprocessing

`LocalImagePreprocessor` قبل از هر API خارجی اجرا می‌شود و فایل خام را overwrite
نمی‌کند. ترتیب پردازش:

1. decode امن، اصلاح EXIF و downscale حافظه‌محور
2. بررسی اجباری وجود انسان با Pose یا detectorهای سبک OpenCV
3. حذف محلی پس‌زمینه شخص و لباس
4. MediaPipe Pose روی CPU
5. Human Parsing اختیاری با ONNX
6. ساخت `replace_mask` و `preserve_mask`
7. crop و سنجش هندسی لباس
8. resize با حفظ نسبت تصویر و letterbox
9. validation امتیازی و ذخیره debug artifacts

اگر person یا garment رد شود، هیچ Vision/LLM یا Try-On API فراخوانی نمی‌شود.
`PERSON_PRESENCE_CHECK_ENABLED=true` مستقل از threshold امتیازی است؛ بنابراین
حتی با `MIN_TRYON_SUITABILITY_SCORE=0`، تصویری که در آن انسان تشخیص داده نشود
با خطای `No person was detected in the person image.` متوقف می‌شود.
Providerهای فعلی GapGPT و OpenRouter دو تصویر normalized را دریافت می‌کنند و
خلاصه بسیار کوتاه pose فقط به prompt تصویری آن‌ها اضافه می‌شود. معماری Provider
قابلیت `supports_mask` دارد؛ ماسک فقط برای Providerی ارسال می‌شود که صریحاً آن
قابلیت را فعال کند.

تنظیمات اصلی:

```env
LOCAL_PREPROCESSING_ENABLED=true
BACKGROUND_REMOVAL_ENABLED=true
POSE_ESTIMATION_ENABLED=false
HUMAN_PARSING_ENABLED=true
HUMAN_PARSING_REQUIRED=false
PREPROCESSING_DEVICE=auto
PREPROCESSING_WARMUP_ENABLED=true
SAVE_PREPROCESSING_DEBUG_IMAGES=false
PREPROCESSING_FAIL_OPEN=false
MIN_TRYON_SUITABILITY_SCORE=0.70
MODEL_CACHE_DIRECTORY=models
LOCAL_MODEL_OFFLINE_MODE=false
```

`PREPROCESSING_WARMUP_ENABLED=true` مدل human parsing را هنگام startup سرویس API
بارگذاری می‌کند تا هزینه‌ی cold start وارد درخواست اول کاربر نشود. پوشه‌ی `models/`
در Compose روی میزبان mount شده است، بنابراین فایل مدل بین restartها باقی می‌ماند.
خاموش بودن `SAVE_PREPROCESSING_DEBUG_IMAGES` فقط مانع ذخیره‌ی تصاویر تشخیصی می‌شود و
هیچ تغییری در تصویر، mask یا prompt ارسالی به مدل تولید ایجاد نمی‌کند.

`PREPROCESSING_DEVICE=auto` ابتدا یک tensor واقعی CUDA می‌سازد و عملیات انجام
می‌دهد. اگر `nvidia-smi`، driver یا tensor CUDA خطا دهد، خطا log و پردازش روی CPU
ادامه پیدا می‌کند. نبود GPU هیچ‌وقت علت crash نیست.

### مدل‌ها و مجوز

- rembg تحت MIT است. مدل‌های `u2net_human_seg` و `u2netp` در اولین استفاده توسط
  rembg داخل `models/rembg` cache می‌شوند و در هر request دوباره load نمی‌شوند.
- Human Parsing پیش‌فرض `schp-atr-18-int8-static.onnx` است؛ حدود 66MB، دارای 18
  کلاس ATR و مجوز MIT. فایل در اولین استفاده از
  `pirocheto/schp-atr-18` دریافت و داخل `models/human_parsing` cache می‌شود.
- مدل‌ها با `.gitignore` از commit خارج‌اند. در حالت
  `LOCAL_MODEL_OFFLINE_MODE=true` نبود مدل پیام خطای واضح می‌دهد؛ اگر parsing
  اجباری نباشد fallback pose/foreground فعال و `degraded_mode=true` ثبت می‌شود.
- `HUMAN_PARSING_MODEL_SHA256` امکان pin کردن checksum مورد تأیید deployment را
  فراهم می‌کند.

مدل SegFormer-B2 رایج عمداً پیش‌فرض نیست، زیرا license مدل پایه NVIDIA استفاده
تجاری را محدود می‌کند.

### اجرای مستقل

PowerShell تک‌خطی:

```powershell
python cli.py preprocess --person "inputs/persons/man.jpg" --garment "inputs/garments/jacket.png" --output "outputs/preprocessing_test"
```

غیرفعال‌کردن Human Parsing:

```powershell
python cli.py preprocess --person "inputs/persons/man.jpg" --garment "inputs/garments/jacket.png" --output "outputs/preprocessing_test" --disable-human-parsing
```

این command هیچ LLM یا Try-On API را صدا نمی‌زند و device، امتیاز، verdict،
هشدارها، علت‌های رد، artifactها و زمان کل را چاپ می‌کند.

## اجرای Mock Mode

مقادیر پیش‌فرض زیر امکان اجرای کاملاً آفلاین را می‌دهند:

```env
USE_MOCK_QWEN=true
USE_MOCK_TRYON=true
```

Mock Try-On تصویر شخص و garment رنگی را ترکیب و برچسب‌گذاری می‌کند و معیار کیفیت
واقعی نیست؛ هدف آن تست wiring، نام‌گذاری، retry و artifactهاست.

## اجرای آزمایشی با OpenRouter

OpenRouter در این پروژه می‌تواند برای دو مسئولیت مستقل استفاده شود:

1. تحلیل شخص، تحلیل لباس و ارزیابی Candidate با یک مدل Vision
2. تولید Candidate با Image API و دو reference image

کلید را از صفحه OpenRouter Keys بسازید و فقط داخل فایل محلی `.env` قرار دهید.
کلید را در کد، Git، Log یا پیام‌ها کپی نکنید:

```env
ANALYSIS_PROVIDER=openrouter
TRYON_PROVIDER=openrouter

OPENROUTER_API_BASE_URL=https://openrouter.ai/api/v1
OPENROUTER_API_KEY=sk-or-v1-your-private-key

# نمونه‌ها هستند؛ مدل انتخابی باید در حساب شما موجود باشد.
OPENROUTER_VISION_MODEL=openai/gpt-4.1-mini
OPENROUTER_IMAGE_MODEL=openai/gpt-image-1

OPENROUTER_TIMEOUT_SECONDS=180
OPENROUTER_HTTP_REFERER=
OPENROUTER_APP_NAME=Virtual Try-On
OPENROUTER_IMAGE_QUALITY=high
OPENROUTER_IMAGE_SIZE=
```

متغیرهای `USE_MOCK_QWEN` و `USE_MOCK_TRYON` در صورت انتخاب صریح Provider نادیده
گرفته می‌شوند. برای استفاده از OpenRouter فقط در تحلیل و نگه‌داشتن تولید Mock:

```env
ANALYSIS_PROVIDER=openrouter
TRYON_PROVIDER=mock
```

پیش از اجرا، تنظیمات را بدون نمایش کلید بررسی کنید:

```powershell
python cli.py config-check
```

سپس ابتدا Vision را جداگانه آزمایش کنید:

```powershell
python cli.py analyze `
  --person inputs/persons/person.jpg `
  --garment inputs/garments/hoodie.png
```

و برای اجرای کامل:

```powershell
python main.py `
  --person inputs/persons/person.jpg `
  --garment inputs/garments/hoodie.png `
  --product-title "Men's hoodie" `
  --candidates-per-color 2
```

مدل Vision باید ورودی تصویر را قبول کند. مدل Image نیز باید در
`GET https://openrouter.ai/api/v1/images/models` دارای ورودی image و قابلیت
`input_references` باشد. OpenRouter client برای سازگاری بیشتر در هر درخواست یک
تصویر تولید می‌کند؛ Pipeline برای Candidateهای بیشتر چند درخواست کنترل‌شده
می‌فرستد.

تولید OpenRouter یک image-to-image عمومی است و Virtual Try-On تخصصی تضمین‌شده
نیست؛ کیفیت حفظ چهره و فرم لباس به مدل Image انتخاب‌شده وابسته است. برای محصول
نهایی همچنان یک Provider تخصصی Virtual Try-On مطمئن‌تر است.

## اجرای GapGPT Platform V2

اشتراک چت GapGPT با اعتبار API متفاوت است. برای API از پنل توسعه‌دهندگان
`/platform-v2`، بخش کلیدها و کیف پول API استفاده کنید. کلید را فقط در `.env`
محلی قرار دهید:

```env
ANALYSIS_PROVIDER=gapgpt
TRYON_PROVIDER=gapgpt

GAPGPT_API_BASE_URL=https://api.gapgpt.app/v1
GAPGPT_API_KEY=your-private-gapgpt-api-key
GAPGPT_VISION_MODEL=gpt-4o
GAPGPT_IMAGE_MODEL=gpt-image-2
GAPGPT_TIMEOUT_SECONDS=180

GAPGPT_IMAGE_EDIT_ENDPOINT=/images/edits
GAPGPT_IMAGE_FIELD_NAME=image[]
GAPGPT_IMAGE_QUALITY=medium
GAPGPT_IMAGE_SIZE=1024x1536
```

تحلیل و ارزیابی از endpoint سازگار با OpenAI یعنی `/chat/completions` استفاده
می‌کند. تولید Try-On دو تصویر person و garment را به‌شکل multipart و با دو field
تکرارشونده `image[]` به `/images/edits` می‌فرستد. پاسخ GapGPT می‌تواند
`b64_json` یا یک URL دانلود HTTPS باشد؛ URL بدون ارسال Authorization دانلود و
قبل از ذخیره به PNG معتبر تبدیل می‌شود.

راهنمای عمومی GapGPT تولید متنی را در `/images/generations` نشان می‌دهد، اما
Virtual Try-On به image editing و تصاویر مرجع نیاز دارد. اگر حساب شما
`/images/edits` را ارائه نکند، برنامه عمداً به text-to-image بدون مرجع fallback
نمی‌کند و خطای روشن 404 می‌دهد. در آن حالت endpoint یا نام field اعلام‌شده توسط
پشتیبانی GapGPT را با `GAPGPT_IMAGE_EDIT_ENDPOINT` و
`GAPGPT_IMAGE_FIELD_NAME` تنظیم کنید.

برای آزمایش کم‌هزینه ابتدا فقط تحلیل را فعال کنید:

```env
ANALYSIS_PROVIDER=gapgpt
TRYON_PROVIDER=mock
```

سپس:

```powershell
python cli.py config-check
python cli.py analyze `
  --person 'inputs/persons/person.jpg' `
  --garment 'inputs/garments/hoodie.png'
```

## CLI

اجرای domain-specific لباس از Task Router:

```powershell
python cli.py clothing `
  --person inputs/persons/person.jpg `
  --garment inputs/garments/hoodie.png `
  --product-title "Men's hoodie"
```

برای ساخت یک استایل از چند تصویر، `--garment` و `--garment-type` را به همان ترتیب
تکرار کنید. هر مرحله خروجی مرحلهٔ قبل را به‌عنوان تصویر شخص دریافت می‌کند:

```bash
python cli.py clothing \
  --person inputs/persons/person.jpg \
  --garment inputs/garments/tshirt.png --garment-type "T-shirt" \
  --garment inputs/garments/pants.png --garment-type "Pants" \
  --garment inputs/accessories/watch.png --garment-type "Watch" \
  --candidates-per-color 2 \
  --max-retries 1
```

حداکثر ۸ تصویر قابل ارسال است. مراحل میانی با یک candidate و بدون retry اجرا
می‌شوند و تعداد candidate و retry درخواستی به مرحلهٔ آخر اختصاص می‌یابد. حالت
چندلباسی رنگ اصلی هر آیتم را حفظ می‌کند.

دستور قدیمی کاملاً سازگار باقی مانده و همان ClothingPipeline را اجرا می‌کند:

```bash
python cli.py run \
  --person inputs/persons/person.jpg \
  --garment inputs/garments/hoodie.png \
  --product-title "Men's hoodie"
```

انتخاب tenant محلی اختیاری است:

```bash
python cli.py clothing --tenant fashion_company \
  --person inputs/persons/person.jpg \
  --garment inputs/garments/hoodie.png \
  --product-title "Men's hoodie"

python cli.py wallpaper --tenant wallpaper_company \
  --room inputs/rooms/room.jpg \
  --wallpaper inputs/wallpapers/reference.png \
  --candidates-per-job 1 \
  --max-retries 1 \
  --pattern-scale 0.18
```

این دستور تحلیل دیوار، تولید mask و texture، فراخوانی generation provider،
ارزیابی Candidate، retry و ساخت `final/wallpaper.png` را انجام می‌دهد. اگر دیوار
مناسب پیدا نشود، status برابر `rejected` می‌شود و Generation API فراخوانی نمی‌شود.

ابزارهای سازگار لباس:

```bash
python cli.py validate \
  --person inputs/persons/person.jpg \
  --garment inputs/garments/hoodie.png

python cli.py analyze \
  --person inputs/persons/person.jpg \
  --garment inputs/garments/hoodie.png

python cli.py preprocess \
  --person inputs/persons/person.jpg \
  --garment inputs/garments/hoodie.png \
  --output outputs/preprocessing_test
```

ورودی JSON نیز پشتیبانی می‌شود:

```bash
python cli.py run --request-json request.json
```

```json
{
  "person_image": "inputs/persons/person.jpg",
  "garment_image": "inputs/garments/hoodie.png",
  "product_title": "Men's hoodie",
  "candidates_per_color": 2,
  "max_retries": 1,
  "preserve_face": true,
  "preserve_pose": true,
  "preserve_background": true
}
```

## Docker

پروژه با دو image اجرا می‌شود: image بک‌اند شامل FastAPI و CLI است و image
فرانت‌اند فقط Streamlit را اجرا می‌کند. از ریشهٔ پروژه اجرا کنید:

```bash
docker compose up --build -d
```

Compose مقدار `API_BASE_URL` فرانت‌اند را در runtime و به‌صورت پیش‌فرض روی
`http://virtual-tryon-backend:8000` قرار می‌دهد. سرویس backend همین network alias
را دارد. این مقدار داخل image فرانت‌اند ذخیره نشده و با `API_BASE_URL` محیط قابل
override است.

پس از healthy شدن سرویس‌ها:

- Frontend: `http://localhost:8501`
- Backend API: `http://localhost:8000`
- API docs: `http://localhost:8000/docs`

برای اجرای standalone imageها روی یک شبکه Docker، آدرس backend باید هنگام اجرای
Frontend تزریق شود؛ در نبود این مقدار، Frontend عمداً با خطای configuration متوقف
می‌شود و به localhost داخل کانتینر fallback نمی‌کند:

```bash
docker network create virtual-tryon-network

docker run --name virtual-tryon-backend \
  --network virtual-tryon-network \
  --env-file .env \
  -p 8000:8000 \
  -v "${PWD}/config:/app/config:ro" \
  -v "${PWD}/outputs:/app/outputs" \
  -v "${PWD}/models:/app/models" \
  cr.samiansoft.com/virtual-tryon-backend:latest

docker run --name virtual-tryon-frontend \
  --network virtual-tryon-network \
  -p 8501:8501 \
  -e API_BASE_URL=http://virtual-tryon-backend:8000 \
  cr.samiansoft.com/virtual-tryon-frontend:latest
```

مشاهدهٔ وضعیت و logها:

```bash
docker compose ps
docker compose logs -f backend frontend
docker compose down
```

اگر UI وضعیت `Backend offline` نشان داد، ابتدا Environment و ارتباط واقعی داخل
کانتینر Frontend را بررسی کنید. image فرانت‌اند `httpx` دارد و به `curl` نیاز نیست:

```bash
docker compose exec frontend python -c \
  "import os; print(os.environ.get('API_BASE_URL'))"

docker compose exec frontend python -c \
  "import os,httpx; u=os.environ['API_BASE_URL'].rstrip('/'); print(httpx.get(u + '/health', timeout=5, trust_env=False).json())"

docker network inspect virtual-tryon-network
docker logs virtual-tryon-backend
```

خروجی فرمان دوم باید JSON شامل `"status": "ok"` باشد. اگر hostname resolve نشد،
هر دو کانتینر روی یک user-defined network نیستند یا نام backend با مقدار
`API_BASE_URL` تطابق ندارد. اگر `Connection refused` دریافت شد، Backend هنوز آماده
نیست یا روی `0.0.0.0:8000` گوش نمی‌دهد. دکمه `Retry backend connection` در Sidebar
نیز health check واقعی را دوباره اجرا می‌کند.

تغییر source code به‌صورت خودکار tag موجود در registry را به‌روزرسانی نمی‌کند.
پس از تغییرات باید هر دو image دوباره build و push و روی میزبان pull شوند:

```bash
docker build -t cr.samiansoft.com/virtual-tryon-backend:latest .
docker build -f frontend/Dockerfile \
  -t cr.samiansoft.com/virtual-tryon-frontend:latest .
docker push cr.samiansoft.com/virtual-tryon-backend:latest
docker push cr.samiansoft.com/virtual-tryon-frontend:latest
docker pull cr.samiansoft.com/virtual-tryon-backend:latest
docker pull cr.samiansoft.com/virtual-tryon-frontend:latest
docker compose up -d --no-build
```

به‌صورت پیش‌فرض providerهای mock فعال‌اند. Docker Compose متغیرهای provider را
از فایل `.env` ریشه برای جای‌گذاری می‌خواند؛ بنابراین کلیدها داخل image ذخیره
نمی‌شوند. برای provider واقعی، متغیرهایی مثل `ANALYSIS_PROVIDER`،
`TRYON_PROVIDER`، `GAPGPT_API_KEY` یا `OPENROUTER_API_KEY` را در `.env` تنظیم
کنید. فایل `config/tenants.json` نیز read-only داخل کانتینر mount می‌شود.

دایرکتوری‌های زیر bind mount هستند و با حذف کانتینر باقی می‌مانند:

```text
inputs/   outputs/   models/   logs/   temp/   config/
```

اولین اجرای preprocessing ممکن است مدل‌ها را دانلود کند و آن‌ها را در `models/`
نگه دارد. image پیش‌فرض وابستگی‌های CPU preprocessing را نصب می‌کند. برای image
سبک‌تر، این دو مقدار را در `.env` قرار دهید و دوباره build کنید:

```dotenv
INSTALL_PREPROCESSING=false
LOCAL_PREPROCESSING_ENABLED=false
```

در image بهینه‌ی Compose، چون pose و background removal در تنظیمات فعلی خاموش‌اند،
وابستگی‌های سنگین MediaPipe و rembg نیز نصب نمی‌شوند. Human parsing و wall
segmentation همچنان از ONNX Runtime استفاده می‌کنند. برای ساخت image کامل با
قابلیت فعال‌سازی این دو feature، مقادیر زیر را پیش از build در `.env` قرار دهید:

```dotenv
INSTALL_POSE=true
INSTALL_BACKGROUND_REMOVAL=true
```

نصب dependencyهای Docker دارای timeout صدوبیست‌ثانیه‌ای، ده retry داخلی pip و
سه تلاش برای هر گروه requirements است؛ cache دانلودها نیز بین buildها حفظ می‌شود.

### اجرای CLI با Docker

CLI داخل همان image بک‌اند قرار دارد و نیازی به سرویس جدا ندارد. یک شخص و چند
لباس/اکسسوری را می‌توان با فرمان یک‌باره اجرا کرد:

```bash
docker compose run --rm backend python cli.py clothing \
  --person inputs/persons/person.jpg \
  --garment inputs/garments/tshirt.png --garment-type "T-shirt" \
  --garment inputs/garments/pants.png --garment-type "Pants" \
  --garment inputs/garments/watch.png --garment-type "Watch" \
  --candidates-per-color 2 \
  --max-retries 1
```

خروجی‌های CLI مستقیماً در `outputs/` میزبان نوشته می‌شوند. سایر فرمان‌ها نیز به
همین شکل قابل اجرا هستند:

```bash
docker compose run --rm backend python cli.py config-check
docker compose run --rm backend python cli.py validate \
  --person inputs/persons/person.jpg \
  --garment inputs/garments/tshirt.png
```

در Linux سرویس backend به‌صورت پیش‌فرض با UID/GID برابر `1000:1000` اجرا می‌شود
تا bind mountها قابل نوشتن باشند. اگر شناسهٔ کاربر میزبان متفاوت است، مقادیر زیر
را در `.env` تنظیم کنید. روی Docker Desktop ویندوز معمولاً این کار لازم نیست.

```dotenv
HOST_UID=1000
HOST_GID=1000
```

## FastAPI

```bash
uvicorn api:app --reload
```

Endpointها:

- `POST /api/v1/generate`: endpoint عمومی با `source_image`،
  `reference_image` و `options` به‌شکل JSON object؛ فیلد `task_type` پذیرفته
  نمی‌شود.
- `POST /api/v1/tryon`: multipart با `person_image`، یک یا چند فیلد تکرارشوندهٔ
  `garment_images` و آرایهٔ JSON متناظر `garment_types`. هر آیتم به‌ترتیب روی
  خروجی قبلی اعمال می‌شود. قرارداد قدیمی `garment_image` و `product_title` برای
  درخواست تک‌لباس همچنان پشتیبانی می‌شود.
- `POST /v1/preprocess`: فقط preprocessing محلی با `person_image`،
  `garment_image` و `human_parsing_enabled` اختیاری؛ هیچ API خارجی فراخوانی
  نمی‌شود و پاسخ فقط نام نسبی artifactها را برمی‌گرداند.
- `GET /api/v1/jobs/{job_id}`
- `GET /api/v1/jobs/{job_id}/results`
- `GET /health`

همهٔ endpointهای Job/Generation، در حالت
`TENANT_AUTH_REQUIRED=true`، کلید tenant را از یکی از headerهای زیر می‌گیرند:

```text
X-API-Key: tenant-secret
Authorization: Bearer tenant-secret
```

نمونهٔ endpoint عمومی؛ pipeline از API key تعیین می‌شود:

```bash
curl -X POST http://127.0.0.1:8000/api/v1/generate \
  -H "X-API-Key: tenant-secret" \
  -F "source_image=@inputs/persons/person.jpg" \
  -F "reference_image=@inputs/garments/hoodie.png" \
  -F 'options={"product_title":"Men hoodie","candidates_per_color":1,"max_retries":0}'
```

رنگ اصلی همهٔ محصولات مرجع به‌صورت پیش‌فرض حفظ می‌شود. برای هر عکس باید دقیقاً
مشخص شود کدام لباس یا اکسسوری از آن منتقل شود:

```bash
curl -X POST http://127.0.0.1:8000/api/v1/tryon \
  -F "person_image=@inputs/persons/person.jpg" \
  -F "garment_images=@inputs/garments/tshirt.png" \
  -F "garment_images=@inputs/garments/pants.png" \
  -F "garment_images=@inputs/accessories/watch.png" \
  -F 'garment_types=["T-shirt","Pants","Watch"]' \
  -F "candidates_per_color=2" -F "max_retries=1"
```

در حالت چندلباسی، برای کاهش هزینه مراحل میانی با یک candidate و بدون retry اجرا
می‌شوند و تنظیمات `candidates_per_color` و `max_retries` روی مرحلهٔ آخر اعمال
می‌شوند. در این حالت فقط رنگ اصلی پشتیبانی می‌شود.

نسخه فعلی job را synchronously اجرا می‌کند. ساخت Pipeline از transport مستقل است،
بنابراین endpoint بعداً می‌تواند فقط Job را در Queue قرار دهد.

## اتصال Qwen API

Qwen client با APIهای OpenAI-compatible کار می‌کند:

```env
USE_MOCK_QWEN=false
QWEN_API_BASE_URL=https://provider.example/v1
QWEN_API_KEY=...
QWEN_MODEL=qwen-vl-model-name
QWEN_TIMEOUT_SECONDS=60
```

اگر base URL به `/chat/completions` ختم نشود، client آن مسیر را اضافه می‌کند.
تصاویر به data URL تبدیل می‌شوند و همراه `response_format=json_object` ارسال
می‌شوند. ابتدا JSON مستقیم parse می‌شود، سپس نخستین object معتبر استخراج و در
نهایت با مدل Pydantic مربوطه validation می‌شود. خروجی خام مدل هیچ‌گاه مصرف
نمی‌شود. پاسخ نامعتبر retry محدود دارد و خطای شبکه/429/5xx دارای exponential
backoff است.

درخواست‌های Vision کاملاً stateless هستند: پیام `system` یا تاریخچه گفتگو ارسال
نمی‌شود و هر درخواست فقط شامل یک پیام `user` فشرده و تصاویر لازم است. prompt
تولید تصویر نیز میان Providerها مشترک و کوتاه نگه داشته شده است.

## اتصال Virtual Try-On API

```env
USE_MOCK_TRYON=false
TRYON_API_BASE_URL=https://provider.example/v1/tryon
TRYON_API_KEY=...
TRYON_MODEL=provider-model-name
TRYON_TIMEOUT_SECONDS=180
```

نسخه اولیه دو فایل را با نام‌های `person_image` و `garment_image` به‌شکل multipart
و options را به‌شکل form fields می‌فرستد. response می‌تواند یک image مستقیم یا
JSON شامل `images`، `outputs` یا `data` با base64 باشد.

برای Providerهایی مانند FASHN، Google یا سرویس داخلی، یک subclass از
`GenericRESTAdapter` بسازید و دقیقاً این دو متد را override کنید:

- `build_multipart(...)`: payload، نام فیلدها، endpoint metadata و گزینه‌های Provider
- `parse_response(...)`: envelope و encoding پاسخ Provider

سپس adapter را به `GenericRESTTryOnClient(adapter=...)` تزریق کنید. اگر Provider
polling، job submission یا URL دانلود دارد، یک پیاده‌سازی مستقل از
`TryOnAPIClient` مناسب‌تر است. این بخش نیازمند قرارداد همان Provider است، اما
interface و محل adaptation کامل و مشخص است.

## Retry

Retry شبکه فقط برای timeout، خطای شبکه، HTTP 429 و 5xx انجام می‌شود. خطاهای 4xx
غیر از rate limit retry نمی‌شوند. اگر بهترین Candidate کمتر از
`MIN_ACCEPTANCE_SCORE` باشد، تا سقف `max_retries` تولید دوباره با گزینه‌های
محافظه‌کارانه زیر انجام می‌شود:

```json
{
  "preserve_face": true,
  "preserve_pose": true,
  "preserve_background": true,
  "strict_identity_preservation": true,
  "only_replace_garment_region": true
}
```

اگر نتیجه همچنان ضعیف باشد، بهترین خروجی موجود با `accepted=false` و وضعیت Job
برابر `completed_with_failures` ذخیره می‌شود.

## تست‌ها

```bash
pytest -v
```

تست‌ها ورودی ناموجود/خراب/کوچک، MIME واقعی، رنگ Hex، تغییر رنگ‌های عادی/سفید/مشکی،
ثابت ماندن خارج mask، وزن‌دهی، threshold و اجرای کامل Mock Pipeline را پوشش
می‌دهند و اینترنت لازم ندارند. مدل‌های preprocessing در تست‌ها mock می‌شوند؛
تست‌های CUDA fallback، EXIF، نسبت تصویر، schema pose، human parsing fallback،
حفاظت face/hands، لباس چسبیده به لبه، singleton مدل، path containment و توقف API
روی validation ردشده نیز وجود دارند. تست‌های معماری tenant mapping، API-key
authentication، Prompt/Model Router، سازگاری کامل ClothingPipeline، endpoint
عمومی و extension pointهای WallpaperPipeline را نیز پوشش می‌دهند.

## Logging و مدیریت خطا

Logها JSON line هستند و شامل Job ID، stage، device، model name، input/output size،
زمان، fallback، degraded mode، تعداد Candidate، retry، score و انتخاب نهایی‌اند.
API key، مسیر حساس، base64، byte تصویر و payload حساس log نمی‌شوند. exceptionهای
دامنه‌ای پیام روشن دارند؛ traceback فقط از logger و متناسب با سطح اجرای برنامه
قابل مشاهده است.

## حریم خصوصی و امنیت

- signature واقعی تصویر با Pillow بررسی می‌شود و صرفاً به پسوند اعتماد نمی‌شود.
- حجم، حداقل/حداکثر ابعاد، نسبت تصویر و فرمت محدود است.
- `..` در مسیر ورودی رد و job lookup داخل output root محصور می‌شود.
- نام Upload و فایل temporary نادیده گرفته شده و نام تصادفی ساخته می‌شود.
- EXIF و metadata در artifactهای پردازش‌شده حذف می‌شود.
- کلید API فقط از environment خوانده می‌شود و محتوای تصویر log نمی‌شود.
- کلید API tenant فقط به‌صورت SHA-256 در config نگهداری و با مقایسه
  constant-time بررسی می‌شود.
- tenant از header احراز هویت resolve می‌شود؛ `task_type` ورودی پذیرفته نمی‌شود.
- نتایج Job متعلق به tenant دیگر با پاسخ 404 پنهان می‌شوند.
- uploadهای API و temp پردازش پس از پایان حذف می‌شوند.
- کد هیچ مسیر آموزش یا نگهداری خارج از artifactهای صریح Job ندارد؛ تصاویر نباید
  بدون قرارداد پردازش داده مناسب به Provider ثالث ارسال شوند.

## محدودیت‌ها

- کیفیت Try-On واقعی کاملاً به Provider انتخابی وابسته است.
- fallback رنگ پس‌زمینه برای تصاویر لباس با پس‌زمینه شلوغ از `rembg` ضعیف‌تر است.
- عکس نیم‌تنه‌ای که هر دو شانه و لگن آن دیده نشود ممکن است validation را رد کند.
- دست‌های ضربدری یا پنهان دقت replace mask را کاهش می‌دهند و امتیاز را کم می‌کنند.
- اگر لباس از قبل با دست، شال، مو یا کت دیگری پوشانده شده باشد، Human Parsing
  ممکن است مرز لباس فعلی را کامل بازیابی نکند.
- fallback بدون مدل Human Parsing فقط از foreground و pose استفاده می‌کند و با
  `degraded_mode=true` مشخص می‌شود؛ برای mask دقیق production مدل ONNX را نصب و
  cache کنید.
- MediaPipe legacy Pose یک pose اصلی را برمی‌گرداند؛ foreground چندتکه و تصاویر
  چندنفره ممکن است برای شمارش دقیق به detector چندنفره جداگانه نیاز داشته باشند.
- recoloring بافت را حفظ می‌کند، اما logo/pattern هم‌رنگ با پارچه ممکن است تا حدی
  تغییر کند.
- API فعلی queue، Redis و retention scheduler ندارد؛ authentication داخلی
  API-key دارد، اما rotation/revocation مرکزی هنوز نیازمند secret manager است.
- کیفیت انتخاب polygon دیوار و حفظ furniture/پنجره‌های داخل آن به Vision و
  Generation provider وابسته است. post-processing پیکسل‌های خارج mask دیوار را
  از تصویر اصلی بازمی‌گرداند، اما occlusionهای داخل خود دیوار باید توسط provider
  حفظ شوند.
- Generic REST Adapter نمی‌تواند قرارداد اختصاصی Provider ناشناخته را حدس بزند؛
  دو نقطه adaptation بالا برای همان قرارداد در نظر گرفته شده‌اند.

## افزودن Domain یا Provider جدید

برای Domain جدید:

1. یک `BasePipeline` و validator دامنه‌ای ایجاد کنید.
2. PromptBuilder آن را در `app/prompts` قرار دهید و در `PromptRouter` ثبت کنید.
3. factory دامنه را در `TaskRouter` ثبت کنید.
4. tenant را با `pipeline` جدید پیکربندی کنید؛ FastAPI تغییر نمی‌کند.

برای Provider جدید:

1. interface تحلیل یا generation مناسب را پیاده کنید.
2. transport، authentication، polling و parsing را در client نگه دارید.
3. construction را به `app/providers/factory.py` اضافه کنید.
4. `ModelRouter` را برای نام مدل آن provider گسترش دهید.
5. تست contract و خطاهای transient/non-transient اضافه کنید؛ endpoint و business
   pipeline نباید تغییر کنند.

## رابط نمایشی Streamlit

رابط حرفه‌ای جلسه مشتری در پوشه `frontend/` قرار دارد و فقط از API عمومی backend
استفاده می‌کند. راهنمای نصب، تنظیم tenant keyها و اجرای دو سرویس در
`frontend/README.md` موجود است.
