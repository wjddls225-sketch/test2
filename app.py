"""AI 관상가 - 얼굴 사진으로 보는 재미용 관상 풀이 (Streamlit)."""

import base64
import hashlib
import html
import io
import os
import re
import warnings

import openai
import streamlit as st
from dotenv import load_dotenv
from PIL import Image, ImageOps, UnidentifiedImageError

load_dotenv()

# gpt-4o / gpt-4o-mini는 인물 사진(명찰·유명인 등)에서 관상 풀이를 거부하는 경우가 많아 4.1-mini를 기본값으로 사용
DEFAULT_MODEL = "gpt-4.1-mini"
MAX_FILE_BYTES = 5 * 1024 * 1024
# 파일 크기는 작아도 픽셀 수가 거대한 이미지(압축 폭탄)를 막기 위한 한도. 약 40MP = 최신 폰 카메라 수준
MAX_IMAGE_PIXELS = 40_000_000
MAX_IMAGE_SIDE = 1024
MAX_OUTPUT_TOKENS = 1500
API_TIMEOUT_SEC = 60
# 공개 배포 시 API 비용 보호: 브라우저 세션(탭)당 분석 횟수 제한
MAX_ANALYSES_PER_SESSION = 5

# Pillow의 기본 한도(약 179MP)보다 낮춰 서버 메모리를 보호한다 (초과 시 DecompressionBombError)
Image.MAX_IMAGE_PIXELS = MAX_IMAGE_PIXELS
# 한도~2배 구간에서 Pillow가 내는 경고는 preprocess_image에서 직접 크기를 검사해 거절하므로 끈다
warnings.simplefilter("ignore", Image.DecompressionBombWarning)

SYSTEM_PROMPT = """당신은 30년 경력의 전통 관상학 전문가 캐릭터입니다.
이 서비스는 사용자가 자기 사진(또는 동의를 받은 사진)을 올려 재미로 보는 엔터테인먼트 관상 풀이입니다.
사용자도 과학적 근거가 없는 놀이라는 것을 알고 있습니다.
사진에서 보이는 얼굴형, 이마, 눈썹, 눈, 코, 입, 귀, 턱의 모양을 전통 관상학의 해석법에 따라
재미와 덕담 위주로 풀이해 주세요.

[원칙]
- 반드시 한국어로, 친근하고 긍정적인 어조로 답합니다.
- 인물이 누구인지 추정하거나 언급하지 않습니다. 옷, 배경, 소품, 명찰 등은 풀이에 사용하지 않고 얼굴 생김새만 봅니다.
- 인종·민족, 외모 비하(못생김 등), 질병 진단은 언급하지 않습니다.
- 사람 얼굴이 사진에 조금이라도 보이면 반드시 아래 형식으로 풀이합니다.
- 오직 사진에 실제 사람의 얼굴이 전혀 없을 때(풍경, 사물, 단색 이미지, 그림·캐릭터 등)에만
  아래 형식을 쓰지 말고 "얼굴을 확인하기 어렵습니다."로 시작해 정면 얼굴 사진을 다시 올려달라고 짧게 안내합니다.
- 여러 사람이 있으면 가장 크게 보이는 한 사람을 기준으로 풀이하고, 첫 줄에 그 사실을 알립니다.

[출력 형식] 아래 마크다운 형식을 정확히 지킵니다.
### 전체 인상
(얼굴형과 첫인상 요약)

### 부위별 풀이
- **이마**: ...
- **눈썹**: ...
- **눈**: ...
- **코**: ...
- **입**: ...
- **귀**: ...
- **턱**: ...
(사진에서 보이지 않는 부위는 생략합니다)

### 성격과 기질
...

### 재물운 / 직업운
...

### 연애운 / 대인관계
...

### 건강운
(가벼운 덕담 수준으로)

### 한 줄 총평
(한 줄 총평과 행운의 조언)
"""

USER_PROMPT = "이 사진 속 인물의 관상을 봐주세요."

FACE_NOT_FOUND_PREFIX = "얼굴을 확인하기 어렵습니다"

SECTION_ICONS = {
    "전체 인상": "🧭",
    "부위별 풀이": "👁️",
    "성격과 기질": "🌿",
    "재물운": "💰",
    "연애운": "💞",
    "건강운": "🍀",
    "한 줄 총평": "✨",
}
SUMMARY_TITLE = "한 줄 총평"

# 테마 색은 .streamlit/config.toml에서 관리하고, 여기서는 직접 만든 HTML 요소만 꾸민다.
# 배경은 반투명 붉은색 + 글자색 상속을 써서 라이트/다크 모드 모두에서 읽히게 한다.
CUSTOM_CSS = """
<style>
.block-container { padding-top: 2.5rem; padding-bottom: 3rem; }

.hero { text-align: center; margin-bottom: 0.5rem; }
.hero-seal {
    display: inline-flex; align-items: center; justify-content: center;
    width: 3.6rem; height: 3.6rem; border-radius: 0.5rem;
    background: #B23A2E; color: #FBF6EA;
    font-family: "Noto Serif KR", serif; font-weight: 900; font-size: 2rem;
    transform: rotate(-4deg); box-shadow: 0 2px 6px rgba(0,0,0,0.18);
}
.hero h1 {
    font-family: "Noto Serif KR", serif !important; font-weight: 900 !important;
    margin: 0.6rem 0 0.2rem 0 !important; padding: 0 !important; letter-spacing: 0.02em;
}
.hero-sub { opacity: 0.8; margin: 0.2rem 0 0.7rem 0; }
.badge {
    display: inline-block; padding: 0.2rem 0.8rem; border-radius: 999px;
    border: 1px solid rgba(178,58,46,0.55); background: rgba(178,58,46,0.10);
    font-size: 0.82rem; font-weight: 500;
}

.steps { display: flex; gap: 0.75rem; margin: 1.2rem 0 0.6rem 0; }
.step {
    flex: 1; text-align: center; padding: 0.9rem 0.4rem;
    border: 1px solid rgba(140,120,90,0.35); border-radius: 0.75rem;
}
.step-num {
    display: inline-block; width: 1.7rem; height: 1.7rem; line-height: 1.7rem; border-radius: 50%;
    background: #B23A2E; color: #FBF6EA; font-weight: 700; font-size: 0.85rem;
}
.step-icon { font-size: 1.6rem; margin: 0.3rem 0 0.1rem 0; }
.step-text { font-size: 0.92rem; font-weight: 500; }

.summary-card {
    border-radius: 0.75rem; padding: 1.1rem 1.3rem; margin: 0.4rem 0 0.8rem 0;
    border-left: 5px solid #B23A2E; background: rgba(178,58,46,0.10);
}
.summary-title { font-family: "Noto Serif KR", serif; font-weight: 700; font-size: 1.1rem; margin-bottom: 0.4rem; }
.summary-body { font-size: 1.02rem; line-height: 1.75; }

/* 업로더 기본 안내("10MB per file")는 서버 한도라 실제 한도(5MB)와 달라 숨긴다. 형식·한도는 라벨에 표시 */
[data-testid="stFileUploaderDropzoneInstructions"] { display: none; }

.footer { text-align: center; font-size: 0.8rem; opacity: 0.7; line-height: 1.8; margin-top: 0.5rem; }

@media (max-width: 640px) {
    .block-container { padding-top: 3.5rem; }
    .hero-seal { width: 3rem; height: 3rem; font-size: 1.7rem; }
    .steps { gap: 0.45rem; }
    .step { padding: 0.7rem 0.2rem; }
    .step-icon { font-size: 1.3rem; }
    .step-text { font-size: 0.8rem; }
}
</style>
"""


class ImageError(Exception):
    """사용자에게 보여줄 이미지 관련 오류."""


# ---------------------------------------------------------------------------
# 설정 / API
# ---------------------------------------------------------------------------

def get_config(key: str, default: str | None = None) -> str | None:
    """st.secrets(Streamlit Cloud)를 먼저 확인하고, 없으면 .env/환경변수를 사용한다."""
    try:
        if key in st.secrets:
            return st.secrets[key]
    except Exception:
        # secrets.toml이 없으면 st.secrets 접근 시 예외가 발생한다 → .env 사용
        pass
    return os.getenv(key, default)


@st.cache_resource
def load_client(api_key: str) -> openai.OpenAI:
    """OpenAI 클라이언트를 생성한다 (키별로 1회만 생성)."""
    return openai.OpenAI(api_key=api_key, timeout=API_TIMEOUT_SEC, max_retries=1)


def preprocess_image(image_bytes: bytes) -> str:
    """이미지를 회전 보정·리사이즈 후 JPEG base64 data URL로 변환한다 (메모리에서만 처리)."""
    too_large_msg = "이미지 해상도가 너무 큽니다. 일반 카메라나 휴대폰으로 찍은 사진을 올려주세요."
    try:
        # Image.open은 헤더만 읽으므로, 픽셀 데이터를 메모리에 펼치기 전에 크기를 검사할 수 있다
        img = Image.open(io.BytesIO(image_bytes))
        width, height = img.size
        if width * height > MAX_IMAGE_PIXELS:
            raise ImageError(too_large_msg)
        # JPEG은 디코딩 단계에서 바로 축소해 메모리 사용을 줄인다
        img.draft("RGB", (MAX_IMAGE_SIDE, MAX_IMAGE_SIDE))
        img = ImageOps.exif_transpose(img)

        if img.mode in ("RGBA", "LA") or (img.mode == "P" and "transparency" in img.info):
            # 투명 배경은 흰색으로 채운다 (그대로 RGB 변환하면 검게 변할 수 있음)
            img = img.convert("RGBA")
            background = Image.new("RGB", img.size, (255, 255, 255))
            background.paste(img, mask=img.getchannel("A"))
            img = background
        elif img.mode != "RGB":
            img = img.convert("RGB")

        img.thumbnail((MAX_IMAGE_SIDE, MAX_IMAGE_SIDE), Image.Resampling.LANCZOS)

        buffer = io.BytesIO()
        img.save(buffer, format="JPEG", quality=85)
    except ImageError:
        raise
    except Image.DecompressionBombError as e:
        raise ImageError(too_large_msg) from e
    except (UnidentifiedImageError, OSError, ValueError, SyntaxError) as e:
        # 손상·위장 파일 (Pillow는 일부 손상 파일에 SyntaxError/ValueError를 낸다)
        raise ImageError("이미지 파일을 읽을 수 없습니다. JPG, PNG, WEBP 형식의 사진을 올려주세요.") from e
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/jpeg;base64,{encoded}"


def analyze_face(client: openai.OpenAI, model: str, image_data_url: str) -> str:
    """OpenAI 이미지 분석 모델로 관상 풀이를 요청한다."""
    response = client.chat.completions.create(
        model=model,
        max_completion_tokens=MAX_OUTPUT_TOKENS,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": USER_PROMPT},
                    {"type": "image_url", "image_url": {"url": image_data_url}},
                ],
            },
        ],
    )
    content = response.choices[0].message.content
    if not content:
        raise ValueError("빈 응답")
    return content.strip()


def friendly_error_message(error: Exception) -> str:
    """예외를 사용자용 한국어 메시지로 바꾼다.

    배포된 앱의 사용자는 설정을 고칠 수 없으므로 설정 관련 오류는 일반 문구로 안내하고,
    자세한 원인은 서버 로그(콘솔)에 남긴다. API 키 등 민감 정보는 포함하지 않는다.
    """
    if isinstance(error, openai.AuthenticationError):
        return "서비스 인증 설정에 문제가 있어 지금은 관상을 볼 수 없습니다. 잠시 후 다시 시도해 주세요."
    if isinstance(error, openai.PermissionDeniedError):
        return "서비스 접근 권한 문제로 지금은 관상을 볼 수 없습니다. 잠시 후 다시 시도해 주세요."
    if isinstance(error, openai.NotFoundError):
        return "분석 모델 설정에 문제가 있어 지금은 관상을 볼 수 없습니다. 잠시 후 다시 시도해 주세요."
    if isinstance(error, openai.InternalServerError):
        return "OpenAI 서버에 일시적인 문제가 있습니다. 잠시 후 다시 시도해 주세요."
    if isinstance(error, openai.RateLimitError):
        return "요청이 너무 많거나 API 사용 한도를 초과했습니다. 잠시 후 다시 시도해 주세요."
    if isinstance(error, openai.APITimeoutError):
        return "분석 시간이 너무 오래 걸려 중단되었습니다. 다시 시도해 주세요."
    if isinstance(error, openai.APIConnectionError):
        return "OpenAI 서버에 연결할 수 없습니다. 인터넷 연결을 확인하고 다시 시도해 주세요."
    if isinstance(error, openai.BadRequestError):
        return "요청을 처리할 수 없습니다. 다른 사진으로 시도하거나, 설정된 모델이 이미지 분석을 지원하는지 확인해 주세요."
    return "관상 분석 중 알 수 없는 오류가 발생했습니다. 잠시 후 다시 시도해 주세요."


# ---------------------------------------------------------------------------
# 상태 관리
# ---------------------------------------------------------------------------

def init_state() -> None:
    defaults = {
        "image_hash": None,   # 현재 결과가 어떤 이미지의 것인지 (sha256)
        "result": None,       # 관상 풀이 마크다운
        "error": None,        # 마지막 분석 오류 메시지
        "analyzing": False,   # 분석 중 버튼 중복 클릭 방지
        "uploader_key": 0,    # 값을 바꾸면 업로드/카메라 위젯이 초기화됨
        "analysis_count": 0,  # 이 세션에서 API를 호출한 횟수 (다시 보기로도 초기화되지 않음)
    }
    for key, value in defaults.items():
        st.session_state.setdefault(key, value)


def clear_result() -> None:
    st.session_state.image_hash = None
    st.session_state.result = None
    st.session_state.error = None
    st.session_state.analyzing = False


def start_analysis() -> None:
    st.session_state.analyzing = True
    st.session_state.error = None


def reset_all() -> None:
    """[다시 보기]: 결과와 업로드한 사진을 모두 비운다."""
    clear_result()
    st.session_state.uploader_key += 1


# ---------------------------------------------------------------------------
# 화면 구성
# ---------------------------------------------------------------------------

def render_header() -> None:
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)
    st.markdown(
        """
        <div class="hero">
            <div class="hero-seal">相</div>
            <h1>AI 관상가</h1>
            <p class="hero-sub">얼굴 사진 한 장으로 보는 전통 관상 풀이</p>
            <span class="badge">🎐 재미로 보는 관상</span>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_steps() -> None:
    # st.columns는 모바일에서 세로로 쌓이므로, 항상 가로 3칸을 유지하도록 HTML flex로 그린다
    steps = [("📸", "사진 올리기"), ("🔮", "관상 보기"), ("📜", "결과 확인")]
    items = "".join(
        f'<div class="step"><span class="step-num">{i}</span>'
        f'<div class="step-icon">{icon}</div><div class="step-text">{text}</div></div>'
        for i, (icon, text) in enumerate(steps, start=1)
    )
    st.markdown(f'<div class="steps">{items}</div>', unsafe_allow_html=True)


def get_image_input():
    """업로드 또는 카메라 촬영으로 이미지를 받는다."""
    st.subheader("사진 올리기")
    source = st.radio(
        "사진 입력 방법",
        ["📁 사진 업로드", "📷 카메라 촬영"],
        horizontal=True,
        label_visibility="collapsed",
    )
    with st.expander("💡 잘 나오는 사진 팁"):
        st.markdown(
            "- 얼굴이 **정면**으로 크게 나온 사진\n"
            "- **밝은 조명**에서 그림자 없이\n"
            "- 사진 속에 **한 명만**\n"
            "- **안경·마스크·모자**는 벗고, 앞머리가 이마를 덮지 않게"
        )
    key = st.session_state.uploader_key
    if source == "📁 사진 업로드":
        return st.file_uploader(
            "얼굴 사진을 올려주세요 (JPG, PNG, WEBP · 최대 5MB)",
            type=["jpg", "jpeg", "png", "webp"],
            key=f"uploader_{key}",
        )
    return st.camera_input("정면을 바라보고 촬영해 주세요", key=f"camera_{key}")


def split_sections(result: str) -> tuple[str, list[tuple[str, str]]]:
    """'### 제목' 기준으로 결과를 (머리말, [(제목, 본문), ...])로 나눈다."""
    parts = re.split(r"^###\s+(.+?)\s*$", result, flags=re.MULTILINE)
    preamble = parts[0].strip()
    sections = [(parts[i].strip(), parts[i + 1].strip()) for i in range(1, len(parts) - 1, 2)]
    return preamble, sections


def section_icon(title: str) -> str:
    for keyword, icon in SECTION_ICONS.items():
        if keyword in title:
            return icon
    return "🔹"


def render_summary_card(title: str, body: str) -> None:
    plain = body.replace("**", "")
    safe = html.escape(plain).replace("\n", "<br>")
    st.markdown(
        f'<div class="summary-card"><div class="summary-title">{section_icon(title)} {html.escape(title)}</div>'
        f'<div class="summary-body">{safe}</div></div>',
        unsafe_allow_html=True,
    )


def render_result(result: str) -> bool:
    """관상 풀이 결과를 카드 형태로 출력한다. 정상적인 풀이 결과였으면 True를 반환한다."""
    if result.startswith(FACE_NOT_FOUND_PREFIX):
        st.warning(result, icon="🙈")
        return False

    preamble, sections = split_sections(result)
    if not sections:
        # 정해진 형식이 아닌 응답(모델의 풀이 거부 등)은 결과로 보여주지 않는다
        st.warning(
            "이 사진으로는 관상 풀이를 받지 못했습니다. "
            "얼굴이 정면으로 크게 나온 다른 사진으로 다시 시도해 주세요.\n\n"
            f"> {result}",
            icon="🙈",
        )
        return False

    st.subheader("📜 관상 풀이 결과")
    if preamble:
        st.info(preamble)

    for title, body in sections:
        if SUMMARY_TITLE in title:
            render_summary_card(title, body)
            continue
        with st.container(border=True):
            st.markdown(f"#### {section_icon(title)} {title}")
            st.markdown(body)
    return True


def render_result_actions(result: str, downloadable: bool) -> None:
    """[다시 보기] 버튼과, 정상 풀이 결과일 때만 [결과 저장] 버튼을 보여준다."""
    if not downloadable:
        st.button("🔄 다른 사진으로 다시 보기", on_click=reset_all, width="stretch")
        return
    col1, col2 = st.columns(2)
    with col1:
        st.button("🔄 다시 보기", on_click=reset_all, width="stretch")
    with col2:
        st.download_button(
            "💾 결과 저장 (.md)",
            data=f"# AI 관상가 - 관상 풀이 결과\n\n{result}\n\n---\n본 결과는 재미를 위한 것이며 과학적 근거가 없습니다.\n",
            file_name="관상_풀이_결과.md",
            mime="text/markdown",
            width="stretch",
        )


def render_footer() -> None:
    st.divider()
    st.markdown(
        '<div class="footer">'
        "⚠️ 본 결과는 재미를 위한 것이며 과학적 근거가 없습니다.<br>"
        "🔒 업로드한 사진은 저장되지 않으며, 분석을 위해 OpenAI API로만 전송됩니다."
        "</div>",
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# 메인
# ---------------------------------------------------------------------------

def main() -> None:
    st.set_page_config(page_title="AI 관상가", page_icon="🔮", layout="centered")
    render_header()

    api_key = get_config("OPENAI_API_KEY")
    model = get_config("OPENAI_MODEL", DEFAULT_MODEL)

    if not api_key:
        st.warning(
            "OpenAI API 키가 설정되지 않았습니다.\n\n"
            "- **로컬 실행**: 프로젝트 폴더의 `.env` 파일에 `OPENAI_API_KEY=sk-...` 를 추가하세요.\n"
            "- **Streamlit Cloud**: 앱 설정 → **Secrets**에 `OPENAI_API_KEY = \"sk-...\"` 를 추가하세요."
        )
        render_footer()
        st.stop()

    init_state()
    render_steps()

    uploaded = get_image_input()

    if uploaded is None:
        clear_result()
        render_footer()
        return

    if uploaded.size > MAX_FILE_BYTES:
        clear_result()
        st.error(
            f"파일 크기({uploaded.size / 1024 / 1024:.1f}MB)가 5MB를 초과합니다. "
            "더 작은 사진을 올려주세요. (휴대폰 사진은 공유 시 '작게' 또는 '중간 크기'를 선택하면 줄어듭니다)"
        )
        render_footer()
        return

    image_bytes = uploaded.getvalue()
    image_hash = hashlib.sha256(image_bytes).hexdigest()
    if image_hash != st.session_state.image_hash:
        # 새 이미지 → 이전 결과 초기화
        clear_result()
        st.session_state.image_hash = image_hash

    try:
        image_data_url = preprocess_image(image_bytes)
    except ImageError as e:
        st.error(str(e))
        render_footer()
        return
    except Exception as e:  # noqa: BLE001 - 예상 못 한 이미지 오류도 traceback 대신 안내만 표시
        print(f"[preprocess_image error] {type(e).__name__}: {e}")
        st.error("이미지를 처리하는 중 문제가 발생했습니다. 다른 사진으로 시도해 주세요.")
        render_footer()
        return

    _, mid, _ = st.columns([1, 2, 1])
    with mid:
        st.image(image_bytes, caption="업로드한 사진", width="stretch")

    remaining = MAX_ANALYSES_PER_SESSION - st.session_state.analysis_count
    if st.session_state.result is None:
        st.button(
            "🔮 관상 보기",
            type="primary",
            width="stretch",
            on_click=start_analysis,
            disabled=st.session_state.analyzing or remaining <= 0,
        )
        if remaining <= 0:
            st.info(
                f"이번 접속에서 볼 수 있는 관상 {MAX_ANALYSES_PER_SESSION}회를 모두 사용했습니다. "
                "잠시 후 다시 방문해 주세요. 🙏"
            )
        else:
            st.caption(f"이번 접속에서 남은 횟수: {remaining} / {MAX_ANALYSES_PER_SESSION}회")

    if st.session_state.analyzing and remaining > 0:
        with st.spinner("관상가가 얼굴을 꼼꼼히 살펴보고 있습니다..."):
            try:
                # 실패한 호출도 비용/요청이 발생할 수 있으므로 호출 전에 횟수를 차감한다
                st.session_state.analysis_count += 1
                client = load_client(api_key)
                st.session_state.result = analyze_face(client, model, image_data_url)
            except Exception as e:  # noqa: BLE001 - 사용자에게는 친화적 메시지만 노출
                print(f"[analyze_face error] {type(e).__name__}: {e}")
                st.session_state.error = friendly_error_message(e)
            finally:
                st.session_state.analyzing = False
        st.rerun()  # 버튼 비활성화 해제 + 결과 화면으로 전환
    st.session_state.analyzing = False  # 한도 초과 상태에서 눌린 경우 등 정리

    if st.session_state.error:
        st.error(st.session_state.error)

    if st.session_state.result:
        downloadable = render_result(st.session_state.result)
        render_result_actions(st.session_state.result, downloadable)

    render_footer()


if __name__ == "__main__":
    main()
