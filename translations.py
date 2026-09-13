"""Bilingual copy for the Image2Genre UI.

Everything the template renders in a human language lives here, keyed by
language code. `direction` drives the `dir` attribute on <html>; the template
and stylesheet handle the rest of the mirroring through logical CSS
properties.
"""

LANGUAGES = ("en", "ar")
DEFAULT_LANGUAGE = "en"

DIRECTION = {"en": "ltr", "ar": "rtl"}

# One signature hue per genre, matched in lightness and chroma so no single
# genre reads louder than the others. Drives --accent on <body>.
GENRE_ACCENTS = {
    "classical": "#8FA6EF",
    "jazz": "#D4A855",
    "rock": "#EF8A6E",
    "pop": "#E88BC8",
    "electronic": "#5AC8E0",
}
FALLBACK_ACCENT = "#8A919B"

GENRE_LABELS = {
    "en": {
        "classical": "Classical",
        "jazz": "Jazz",
        "rock": "Rock",
        "pop": "Pop",
        "electronic": "Electronic",
    },
    "ar": {
        "classical": "كلاسيكية",
        "jazz": "جاز",
        "rock": "روك",
        "pop": "بوب",
        "electronic": "إلكترونية",
    },
}

GENRE_DESCRIPTIONS = {
    "en": {
        "classical": "Calm and inspiring — cool colour, open space, timeless composure.",
        "jazz": "Sophisticated and smooth — warm tone, soft edges, a relaxed room.",
        "rock": "Powerful and raw — bold texture, hard contrast, strong emotion.",
        "pop": "Energetic and vibrant — bright colour, immediate contrast, modern surface.",
        "electronic": "Dynamic and futuristic — pulsating rhythm, abstract geometry, synthetic light.",
    },
    "ar": {
        "classical": "هادئة ومُلهِمة — ألوان باردة، ومساحة مفتوحة، ورزانة لا يطويها زمن.",
        "jazz": "أنيقة وسلسة — نبرة دافئة، وحواف ناعمة، وغرفة مسترخية.",
        "rock": "قوية وخام — ملمس جريء، وتباين حاد، وانفعال شديد.",
        "pop": "نشِطة ونابضة — لون ساطع، وتباين مباشر، وسطح عصري.",
        "electronic": "حيوية ومستقبلية — إيقاع نابض، وأشكال مجرّدة، وضوء اصطناعي.",
    },
}

# Appended to the vision-model prompt so the explanations come back in the
# language the page is being viewed in.
PROMPT_LANGUAGE = {
    "en": "Respond in English.",
    "ar": "أجب بالعربية الفصحى المعاصرة، بأسلوب واضح وموجز.",
}

UI = {
    "en": {
        "page_title": "Image2Genre — find the music in your image",
        "brand": "Image2Genre",
        "model_chip": "ResNet34 · 5 classes",
        "lang_other": "ع",
        "lang_other_label": "Switch to Arabic",
        "new_image": "New image",

        "hero_eyebrow": "Visual vibe → sound",
        "hero_title": "Show me a photo.\nI'll find the\nmusic in it.",
        "hero_body": "A ResNet34 reads the colour, light and texture of your image and picks the genre that matches its mood — then shows you exactly which part of the frame changed its mind.",
        "five_vibes": "Five possible vibes",
        "drop_here": "Drop an image here",
        "or": "or",
        "choose_file": "Choose a file",
        "privacy": "Your image is stored to improve the model. Nothing is shared publicly.",
        "analyze": "Discover music",

        "source": "Source",
        "attention": "Model attention",
        "low": "Low",
        "high": "High",
        "normalized": "224² normalized",

        "predicted_vibe": "Predicted vibe",
        "distribution": "Class distribution",
        "what_image_shows": "What the image shows",
        "why_model_said": "Why the model said so",

        "sounds_like": "Sounds like this",
        "tracks_count": "{count} tracks",
        "tab_global": "Global",
        "tab_khaleeji": "Khaleeji",
        "prev": "Previous",
        "next": "Next",
        "no_playlists": "No playlists found for this genre.",
        "embed_note": "Some videos can't be previewed inline because of YouTube embedding or regional limits — the title links out to YouTube.",
        "watch_on_youtube": "Watch on YouTube",

        "analyzing": "Analyzing",
        "reading_image": "Reading your image",
        "listening": "Listening for\nthe vibe…",
        "step_1": "Image decoded",
        "step_1_sub": "Resized to 224², ImageNet normalization applied",
        "step_2": "Classifying the vibe",
        "step_2_sub": "ResNet34 forward pass across 5 genres",
        "step_3": "Locating attention",
        "step_3_sub": "Grad-CAM over the last convolution block",
        "step_4": "Writing the explanation",
        "step_4_sub": "Vision model describes both frames",
        "heatmap_pending": "Heatmap appears here",

        "error": "Something went wrong",
    },
    "ar": {
        "page_title": "Image2Genre — اكتشف الموسيقى في صورتك",
        "brand": "Image2Genre",
        "model_chip": "ResNet34 · خمس فئات",
        "lang_other": "EN",
        "lang_other_label": "التبديل إلى الإنجليزية",
        "new_image": "صورة جديدة",

        "hero_eyebrow": "من صورة الى اغنية",
        "hero_title": "أرِني صورة.\nسأجد لك\nالموسيقى فيها.",
        "hero_body": "يقرأ نموذج ResNet34 ألوان صورتك وضوءها وملمسها، ويختار النوع الموسيقي الذي يطابق مزاجها — ثم يُريك أي جزء من الصورة كان سبب في رأيه.",
        "five_vibes": "خمسة أنواع موسيقية محتملة",
        "drop_here": "أفلِت صورة هنا",
        "or": "أو",
        "choose_file": "اختر ملفًا",
        "privacy": "تُحفظ صورتك لتحسين النموذج، ولا يُنشر منها شيء علنًا.",
        "analyze": "اكتشف الموسيقى",

        "source": "الصورة الأصلية",
        "attention": "تركيز النموذج",
        "low": "منخفض",
        "high": "مرتفع",
        "normalized": "بُعد الصورة 224²",

        "predicted_vibe": "الفئة المختارة",
        "distribution": "توزيع الفئات",
        "what_image_shows": "ما تُظهره الصورة",
        "why_model_said": "لماذا اختار النموذج ذلك",

        "sounds_like": "موسيقى تناسب الصورة",
        "tracks_count": "{count} مقطعًا",
        "tab_global": "عالمي",
        "tab_khaleeji": "خليجي",
        "prev": "السابق",
        "next": "التالي",
        "no_playlists": "لا توجد قوائم تشغيل لهذا النوع.",
        "embed_note": "بعض المقاطع لا تُعرض داخل الصفحة بسبب قيود التضمين أو القيود الجغرافية في يوتيوب — العنوان يفتحها على يوتيوب.",
        "watch_on_youtube": "شاهد على يوتيوب",

        "analyzing": "جارٍ التحليل",
        "reading_image": "قراءة صورتك",
        "listening": "نستمع إلى\nالطابع…",
        "step_1": "فُكّ ترميز الصورة",
        "step_1_sub": "تغيير الحجم إلى 224² وتنفيذ ImageNet",
        "step_2": "تصنيف الطابع",
        "step_2_sub": "استخدام خوارزمية عبر ResNet34 على 5 فئات",
        "step_3": "تحديد مواضع التركيز",
        "step_3_sub": "Grad-CAM على آخر طبقة في الشبكة العصبية",
        "step_4": "كتابة التفسير",
        "step_4_sub": "نموذج بصري يصف كلتا الصورتين",
        "heatmap_pending": "تظهر الخريطة الحرارية هنا",

        "error": "حدث خطأ ما",
    },
}


def normalize(lang):
    """Coerce an arbitrary lang value to a supported code."""
    return lang if lang in LANGUAGES else DEFAULT_LANGUAGE


def other(lang):
    """The language the toggle switches to."""
    return "ar" if normalize(lang) == "en" else "en"
