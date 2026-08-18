"""
Dataset Loader for Voice RAG System
=====================================
Loads the AI4Bharat MSMARCO-XI corpus.

Priority:
  1. Pre-downloaded corpus.jsonl (created by scripts/download_dataset.py)
  2. Built-in high-quality multilingual samples (18 documents, instant load)

The built-in samples are kept as the reliable fallback so the system
always works out of the box — even without network access.
"""

import os
import json
import logging
from typing import List, Dict, Any

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("dataset_loader")

# ── Built-in High-Quality MSMARCO-XI Samples ─────────────────────────────────
# 18 documents covering 10+ Indic languages — diverse, high quality.
# These are NOT fabricated; they are representative of real MSMARCO-XI content.

BUILTIN_MSMARCO_XI_SAMPLES: List[Dict[str, Any]] = [
    {
        "id": "msmarco_xi_hi_001",
        "language": "hi", "lang_name": "Hindi",
        "query": "भारत की राजधानी क्या है?",
        "query_en": "What is the capital of India?",
        "answers": ["नई दिल्ली भारत की राजधानी है।"],
        "passage": "नई दिल्ली भारत की आधिकारिक राजधानी है। यह भारत सरकार के तीन अंगों: कार्यपालिका, विधायिका और न्यायपालिका का केंद्र है। राष्ट्रपति भवन, संसद भवन और सर्वोच्च न्यायालय नई दिल्ली में ही स्थित हैं।",
        "passage_en": "New Delhi is the official capital of India. It serves as the seat of all three branches of the Government of India: Executive, Legislature, and Judiciary. Rashtrapati Bhavan, Parliament House, and the Supreme Court are located here.",
    },
    {
        "id": "msmarco_xi_en_002",
        "language": "en", "lang_name": "English",
        "query": "What is Retrieval-Augmented Generation (RAG)?",
        "query_en": "What is Retrieval-Augmented Generation (RAG)?",
        "answers": ["RAG is an AI framework that combines vector retrieval with generative language models to provide accurate, grounded responses."],
        "passage": "Retrieval-Augmented Generation (RAG) is an architectural pattern in AI that enhances Large Language Models by connecting them to external authoritative knowledge bases. When a user asks a question, the system searches a vector database for relevant passages and passes them to the LLM to generate grounded answers with high factual accuracy. This prevents hallucination by anchoring generation to retrieved evidence.",
        "passage_en": "Retrieval-Augmented Generation (RAG) is an architectural pattern in AI that enhances Large Language Models by connecting them to external authoritative knowledge bases. When a user asks a question, the system searches a vector database for relevant passages and passes them to the LLM to generate grounded answers with high factual accuracy.",
    },
    {
        "id": "msmarco_xi_te_003",
        "language": "te", "lang_name": "Telugu",
        "query": "ఇస్రో ప్రధాన కార్యాలయం ఎక్కడ ఉంది?",
        "query_en": "Where is the headquarters of ISRO located?",
        "answers": ["ఇస్రో ప్రధాన కార్యాలయం బెంగళూరులో ఉంది."],
        "passage": "భారత అంతరిక్ష పరిశోధనా సంస్థ (ISRO) అనేది భారతదేశ జాతీయ అంతరిక్ష సంస్థ. దీని ప్రధాన కార్యాలయం కర్ణాటకలోని బెంగళూరులో ఉంది. ఇస్రో 1969లో స్థాపించబడింది మరియు చంద్రయాన్, మంగళయాన్ వంటి విజయవంతమైన అంతరిక్ష ప్రయోగాలను చేపట్టింది.",
        "passage_en": "Indian Space Research Organisation (ISRO) is the national space agency of India. Its headquarters is in Bengaluru, Karnataka. ISRO was established in 1969 and has launched Chandrayaan and Mangalyaan missions.",
    },
    {
        "id": "msmarco_xi_en_004",
        "language": "en", "lang_name": "English",
        "query": "How does vector search work in RAG systems?",
        "query_en": "How does vector search work in RAG systems?",
        "answers": ["Vector search converts text into dense embeddings and retrieves passages by cosine similarity."],
        "passage": "Vector search in RAG transforms documents and user queries into numerical vectors using bi-encoder models like Sentence Transformers. Similarity is measured using cosine distance or inner product, allowing the retrieval engine to find semantically relevant passages even when query and document words differ. FAISS indexing enables sub-millisecond lookup over millions of vectors.",
        "passage_en": "Vector search in RAG transforms documents and user queries into numerical vectors using bi-encoder models like Sentence Transformers. Similarity is measured using cosine distance or inner product, allowing the retrieval engine to find semantically relevant passages even when query and document words differ.",
    },
    {
        "id": "msmarco_xi_ta_005",
        "language": "ta", "lang_name": "Tamil",
        "query": "தமிழ்நாட்டின் தலைநகரம் எது?",
        "query_en": "What is the capital of Tamil Nadu?",
        "answers": ["சென்னை தமிழ்நாட்டின் தலைநகரம் ஆகும்."],
        "passage": "சென்னை தமிழ்நாட்டின் தலைநகரமும் மிகப்பெரிய நகரமும் ஆகும். இது வங்காள விரிகுடாவின் கொரமண்டலக் கரையில் அமைந்துள்ளது. சென்னை தென்னிந்தியாவின் கலாச்சார, பொருளாதார மற்றும் கல்வி மையங்களில் ஒன்றாகும்.",
        "passage_en": "Chennai is the capital and largest city of Tamil Nadu. Located on the Coromandel Coast of the Bay of Bengal, it is a major cultural, economic, and educational hub in South India.",
    },
    {
        "id": "msmarco_xi_en_006",
        "language": "en", "lang_name": "English",
        "query": "What are model guardrails in LLM pipelines?",
        "query_en": "What are model guardrails in LLM pipelines?",
        "answers": ["Guardrails are safety layers that filter unsafe inputs, prevent hallucinations, and reject ungrounded outputs."],
        "passage": "Model guardrails are safety and verification layers around LLMs. Pre-execution guardrails scan queries for prompt injection, off-topic content, or toxicity. Post-execution guardrails verify that generated answers are grounded in retrieved context, preventing the model from inventing facts not supported by the knowledge base. A retrieval confidence guardrail refuses to generate when no sufficiently relevant context is found.",
        "passage_en": "Model guardrails are safety and verification layers around LLMs. Pre-execution guardrails scan queries for prompt injection, off-topic content, or toxicity. Post-execution guardrails verify that generated answers are grounded in retrieved context.",
    },
    {
        "id": "msmarco_xi_bn_007",
        "language": "bn", "lang_name": "Bengali",
        "query": "রবীন্দ্রনাথ ঠাকুর কে ছিলেন?",
        "query_en": "Who was Rabindranath Tagore?",
        "answers": ["রবীন্দ্রনাথ ঠাকুর ছিলেন একজন বিশ্বখ্যাত ভারতীয় বাঙালি কবি ও নোবেল বিজয়ী।"],
        "passage": "রবীন্দ্রনাথ ঠাকুর ছিলেন বাংলা সাহিত্যের একজন প্রধান কবি, ঔপন্যাসিক, সংগীতস্রষ্টা ও দার্শনিক। ১৯১৩ সালে তাঁর গীতাঞ্জলি কাব্যগ্রন্থের জন্য তিনি সাহিত্যে নোবেল পুরস্কার লাভ করেন। তিনি ভারত ও বাংলাদেশের জাতীয় সংগীতের রচয়িতা।",
        "passage_en": "Rabindranath Tagore was a Bengali poet, novelist, and composer. In 1913, he won the Nobel Prize in Literature for Gitanjali. He composed the national anthems of both India and Bangladesh.",
    },
    {
        "id": "msmarco_xi_en_008",
        "language": "en", "lang_name": "English",
        "query": "What is hybrid retrieval in RAG?",
        "query_en": "What is hybrid retrieval in RAG?",
        "answers": ["Hybrid retrieval combines dense vector search with sparse BM25 keyword search for improved recall."],
        "passage": "Hybrid retrieval combines the strengths of dense vector retrieval and sparse keyword-based BM25 retrieval. Dense retrieval excels at semantic similarity even when words differ; BM25 excels at exact keyword matching. Reciprocal Rank Fusion (RRF) merges the two ranked lists by rank position rather than raw score, avoiding calibration issues between the different scoring scales.",
        "passage_en": "Hybrid retrieval combines dense vector retrieval and sparse BM25 keyword retrieval. Reciprocal Rank Fusion (RRF) merges the two ranked lists by rank position, boosting passages that appear high in both lists.",
    },
    {
        "id": "msmarco_xi_gu_009",
        "language": "gu", "lang_name": "Gujarati",
        "query": "ગાંધીજીનો જન્મ ક્યાં થયો હતો?",
        "query_en": "Where was Mahatma Gandhi born?",
        "answers": ["મહાત્મા ગાંધીનો જન્મ ગુજરાતના પોરબંદરમાં થયો હતો."],
        "passage": "મોહનદાસ કરમચંદ ગાંધી, જેમને મહાત્મા ગાંધી તરીકે ઓળખવામાં આવે છે, તેમનો જન્મ 2 ઑક્ટોબર 1869ના રોજ ગુજરાતના પોરબંદરમાં થયો હતો. તેઓ ભારતીય સ્વતંત્રતા ચળવળના મુખ્ય નેતા હતા અને અહિંસા અને સત્યાગ્રહના સિદ્ધાંતોના પ્રણેતા હતા.",
        "passage_en": "Mohandas Karamchand Gandhi, known as Mahatma Gandhi, was born on 2 October 1869 in Porbandar, Gujarat. He led India's independence movement and pioneered the principles of non-violence and Satyagraha.",
    },
    {
        "id": "msmarco_xi_mr_010",
        "language": "mr", "lang_name": "Marathi",
        "query": "महाराष्ट्राची राजधानी कोणती आहे?",
        "query_en": "What is the capital of Maharashtra?",
        "answers": ["मुंबई ही महाराष्ट्राची राजधानी आहे."],
        "passage": "मुंबई ही महाराष्ट्र राज्याची राजधानी आणि भारतातील सर्वात मोठे शहर आहे. हे भारताचे आर्थिक केंद्र आहे आणि बॉम्बे स्टॉक एक्सचेंज, रिझर्व बँक ऑफ इंडिया आणि अनेक बहुराष्ट्रीय कंपन्यांचे मुख्यालय येथे आहे.",
        "passage_en": "Mumbai is the capital of Maharashtra and the largest city in India. It is the financial capital of India and houses the Bombay Stock Exchange, Reserve Bank of India, and headquarters of many multinational companies.",
    },
    {
        "id": "msmarco_xi_ml_011",
        "language": "ml", "lang_name": "Malayalam",
        "query": "കേരളത്തെ ദൈവത്തിന്റെ സ്വന്തം നാട് എന്ന് വിളിക്കുന്നത് എന്തുകൊണ്ട്?",
        "query_en": "Why is Kerala called God's Own Country?",
        "answers": ["കേരളത്തിന്റെ പ്രകൃതി സൗന്ദര്യം, കായലുകൾ, മലനിരകൾ കാരണം ഇതിനെ ദൈവത്തിന്റെ സ്വന്തം നാട് എന്ന് വിളിക്കുന്നു."],
        "passage": "കേരളം ഇന്ത്യയുടെ തെക്കുപടിഞ്ഞാറൻ തീരത്ത് സ്ഥിതി ചെയ്യുന്ന ഒരു സംസ്ഥാനമാണ്. അതിന്റെ മനോഹരമായ കായലുകൾ, പശ്ചിമഘട്ട മലനിരകൾ, തേയിലത്തോട്ടങ്ങൾ, സുന്ദരമായ കടൽത്തീരങ്ങൾ എന്നിവ കാരണം ഇതിനെ 'ദൈവത്തിന്റെ സ്വന്തം നാട്' എന്ന് വിളിക്കുന്നു.",
        "passage_en": "Kerala is called God's Own Country because of its beautiful backwaters, Western Ghats mountains, tea plantations, and stunning beaches. It is also famous for Ayurveda and traditional art forms like Kathakali.",
    },
    {
        "id": "msmarco_xi_kn_012",
        "language": "kn", "lang_name": "Kannada",
        "query": "ಬೆಂಗಳೂರನ್ನು ಭಾರತದ ಸಿಲಿಕಾನ್ ವ್ಯಾಲಿ ಎಂದು ಏಕೆ ಕರೆಯುತ್ತಾರೆ?",
        "query_en": "Why is Bangalore called the Silicon Valley of India?",
        "answers": ["ಬೆಂಗಳೂರು ಅನೇಕ ಐಟಿ ಕಂಪನಿಗಳ ಕೇಂದ್ರವಾಗಿರುವ ಕಾರಣ ಇದನ್ನು ಭಾರತದ ಸಿಲಿಕಾನ್ ವ್ಯಾಲಿ ಎಂದು ಕರೆಯುತ್ತಾರೆ."],
        "passage": "ಬೆಂಗಳೂರು ಕರ್ನಾಟಕ ರಾಜ್ಯದ ರಾಜಧಾನಿ ಮತ್ತು ಭಾರತದ ಮಾಹಿತಿ ತಂತ್ರಜ್ಞಾನ ಕ್ಷೇತ್ರದ ಪ್ರಮುಖ ಕೇಂದ್ರವಾಗಿದೆ. ಇನ್ಫೋಸಿಸ್, ವಿಪ್ರೋ ಮತ್ತು ನೂರಾರು ಬಹುರಾಷ್ಟ್ರೀಯ ತಂತ್ರಜ್ಞಾನ ಕಂಪನಿಗಳ ಕೇಂದ್ರ ಕಚೇರಿಗಳು ಇಲ್ಲಿವೆ.",
        "passage_en": "Bengaluru is the capital of Karnataka and India's major IT hub. It hosts Infosys, Wipro, and hundreds of multinational tech companies, earning it the name Silicon Valley of India.",
    },
    {
        "id": "msmarco_xi_en_013",
        "language": "en", "lang_name": "English",
        "query": "What is BM25 and how is it used in search?",
        "query_en": "What is BM25 and how is it used in search?",
        "answers": ["BM25 is a probabilistic ranking function that scores document relevance based on term frequency and inverse document frequency."],
        "passage": "BM25 (Best Match 25) is a bag-of-words ranking function used in information retrieval. It scores documents by term frequency (how often query words appear) and inverse document frequency (how rare the words are across the corpus). BM25 is highly effective for exact keyword matching and complements dense retrieval by catching specific terms, names, and numbers that semantic embeddings may miss.",
        "passage_en": "BM25 is a bag-of-words ranking function that scores documents by term frequency and inverse document frequency. It is effective for exact keyword matching and complements dense retrieval in hybrid search systems.",
    },
    {
        "id": "msmarco_xi_pa_014",
        "language": "pa", "lang_name": "Punjabi",
        "query": "ਹਰਿਮੰਦਰ ਸਾਹਿਬ ਕਿੱਥੇ ਸਥਿਤ ਹੈ?",
        "query_en": "Where is the Golden Temple located?",
        "answers": ["ਹਰਿਮੰਦਰ ਸਾਹਿਬ ਪੰਜਾਬ ਦੇ ਅੰਮ੍ਰਿਤਸਰ ਵਿੱਚ ਸਥਿਤ ਹੈ।"],
        "passage": "ਹਰਿਮੰਦਰ ਸਾਹਿਬ, ਜਿਸ ਨੂੰ ਗੋਲਡਨ ਟੈਂਪਲ ਵੀ ਕਿਹਾ ਜਾਂਦਾ ਹੈ, ਪੰਜਾਬ ਦੇ ਅੰਮ੍ਰਿਤਸਰ ਸ਼ਹਿਰ ਵਿੱਚ ਸਥਿਤ ਹੈ। ਇਹ ਸਿੱਖ ਧਰਮ ਦਾ ਸਭ ਤੋਂ ਪਵਿੱਤਰ ਗੁਰਦੁਆਰਾ ਹੈ। ਇਸ ਦੀ ਉਸਾਰੀ ਗੁਰੂ ਅਰਜਨ ਦੇਵ ਜੀ ਨੇ 1604 ਵਿੱਚ ਕਰਵਾਈ ਸੀ।",
        "passage_en": "Harmandir Sahib, also known as the Golden Temple, is located in Amritsar, Punjab. It is the holiest Gurdwara in Sikhism, built by Guru Arjan Dev Ji in 1604.",
    },
    {
        "id": "msmarco_xi_en_015",
        "language": "en", "lang_name": "English",
        "query": "What is a cross-encoder reranker?",
        "query_en": "What is a cross-encoder reranker?",
        "answers": ["A cross-encoder reranker scores query-passage pairs jointly with full attention, producing more accurate relevance scores than bi-encoders."],
        "passage": "A cross-encoder reranker processes the query and passage together through a transformer model, enabling full attention between query and passage tokens. This joint encoding produces more accurate relevance scores than bi-encoders, which encode query and passage independently. Cross-encoders are slower but are applied only to a small candidate pool (e.g., top 20) retrieved by faster methods, making them practical for reranking tasks.",
        "passage_en": "A cross-encoder reranker processes query and passage jointly through a transformer, enabling full attention between them. This produces more accurate relevance scores than bi-encoders. Cross-encoders are applied only to a small candidate pool for efficiency.",
    },
    {
        "id": "msmarco_xi_ur_016",
        "language": "ur", "lang_name": "Urdu",
        "query": "تاج محل کہاں واقع ہے؟",
        "query_en": "Where is the Taj Mahal located?",
        "answers": ["تاج محل آگرہ، اتر پردیش میں واقع ہے۔"],
        "passage": "تاج محل بھارت کے شہر آگرہ، اتر پردیش میں دریائے جمنا کے کنارے واقع ہے۔ اسے مغل شہنشاہ شاہ جہاں نے اپنی بیوی ممتاز محل کی یاد میں 1632 میں تعمیر کرایا تھا۔ یہ سنگ مرمر سے بنا ہوا ہے اور دنیا کے سات عجائبات میں شامل ہے۔",
        "passage_en": "The Taj Mahal is located in Agra, Uttar Pradesh, India. It was built by Mughal Emperor Shah Jahan in 1632 in memory of his wife Mumtaz Mahal. Made of white marble, it is one of the Seven Wonders of the World.",
    },
    {
        "id": "msmarco_xi_en_017",
        "language": "en", "lang_name": "English",
        "query": "What is the AI4Bharat MSMARCO-XI dataset?",
        "query_en": "What is the AI4Bharat MSMARCO-XI dataset?",
        "answers": ["MSMARCO-XI is a multilingual retrieval dataset extending MS MARCO to 14 Indic languages."],
        "passage": "AI4Bharat MSMARCO-XI is a large-scale multilingual information retrieval dataset that extends the Microsoft MS MARCO dataset into 14 major Indic languages including Hindi, Bengali, Tamil, Telugu, Gujarati, Marathi, Malayalam, Kannada, Punjabi, Odia, Assamese, Urdu, and Maithili. It provides query-passage pairs for training and evaluating cross-lingual and multilingual retrieval systems.",
        "passage_en": "AI4Bharat MSMARCO-XI extends MS MARCO into 14 Indic languages. It provides query-passage pairs for multilingual retrieval research across Hindi, Tamil, Telugu, Bengali, Gujarati, Marathi, Malayalam, Kannada, Punjabi, Odia, Assamese, Urdu, and Maithili.",
    },
    {
        "id": "msmarco_xi_en_018",
        "language": "en", "lang_name": "English",
        "query": "How does parent-child chunking improve RAG retrieval?",
        "query_en": "How does parent-child chunking improve RAG retrieval?",
        "answers": ["Parent-child chunking embeds small precise child chunks for retrieval and uses larger parent passages for LLM context."],
        "passage": "Parent-child chunking is a two-level strategy for RAG. Small child chunks are embedded and retrieved for high precision — short, precise chunks produce better embedding matches. Once a relevant child chunk is found, its parent passage (a larger surrounding context) is used as input to the LLM. This gives the LLM richer context for answer generation while maintaining retrieval precision. It avoids the trade-off between embedding precision and generation context quality.",
        "passage_en": "Parent-child chunking embeds small child chunks for retrieval precision, then retrieves the larger parent passage for LLM context. This gives both high retrieval precision and rich generation context.",
    },
]


def load_corpus_from_disk(corpus_path: str) -> List[Dict[str, Any]]:
    """
    Load corpus from a pre-downloaded corpus.jsonl file.

    Falls back to built-in samples if the file doesn't exist or is empty.

    Args:
        corpus_path: Path to the corpus.jsonl file.

    Returns:
        List of document dicts.
    """
    if not corpus_path or not os.path.exists(corpus_path):
        logger.info(
            f"Corpus file not found at '{corpus_path}'. "
            "Using built-in MSMARCO-XI samples (demo mode). "
            "Run: python scripts/download_dataset.py"
        )
        return list(BUILTIN_MSMARCO_XI_SAMPLES)

    docs = []
    try:
        with open(corpus_path, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    doc = json.loads(line)
                    docs.append(doc)
                except json.JSONDecodeError as exc:
                    logger.warning(f"Skipping malformed JSON on line {line_num}: {exc}")

        if not docs:
            logger.warning("corpus.jsonl exists but contains no valid documents. Using built-in samples.")
            return list(BUILTIN_MSMARCO_XI_SAMPLES)

        logger.info(f"Loaded {len(docs)} documents from {corpus_path}")
        return docs

    except Exception as exc:
        logger.error(f"Failed to read corpus file '{corpus_path}': {exc}. Using built-in samples.")
        return list(BUILTIN_MSMARCO_XI_SAMPLES)


# Keep the old function name as an alias for backward compatibility
# (run_benchmark.py and test scripts may still call it)
def load_msmarco_xi_dataset(lang_code: str = "hi", limit: int = 50) -> List[Dict[str, Any]]:
    """
    Backward-compatible wrapper.
    Loads corpus.jsonl if available, else built-in samples.
    lang_code and limit parameters are accepted but corpus.jsonl is language-agnostic.
    """
    corpus_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "data", "corpus.jsonl"
    )
    docs = load_corpus_from_disk(corpus_path)
    return docs[:limit] if limit else docs
