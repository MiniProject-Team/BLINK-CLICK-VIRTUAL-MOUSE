def normalize(text: str) -> str:
    text = text.lower()

    corrections = {
        "krom": "chrome",
        "crome": "chrome",
        "serch": "search",
        "youtub": "youtube",
    }

    for source, target in corrections.items():
        text = text.replace(source, target)

    return text.strip()
