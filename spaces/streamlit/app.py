
import streamlit as st

from optimum_transformers.pipelines import SUPPORTED_TASKS
from optimum_transformers import Benchmark
from optimum_transformers.utils.benchmark import plot_benchmark

st.set_option('deprecation.showPyplotGlobalUse', False)

st.set_page_config(page_title="Optimum Transformers")

st.title("Optimum Transformers")
st.sidebar.markdown(
    """
<style>
.aligncenter {
    text-align: center;
}
</style>
<p class="aligncenter">
    <img src="https://raw.githubusercontent.com/AlekseyKorshuk/optimum-transformers/master/data/social_preview.png" width="300" />
</p>
""",
    unsafe_allow_html=True,
)
st.sidebar.markdown(
    """
<style>
.aligncenter {
    text-align: center;
}
</style>

<p style='text-align: center'>
<a href="https://github.com/AlekseyKorshuk/optimum-transformers" target="_blank">GitHub</a>
</p>

<p class="aligncenter">
    <a href="https://github.com/AlekseyKorshuk/optimum-transformers" target="_blank"> 
        <img src="https://img.shields.io/github/stars/AlekseyKorshuk/optimum-transformers?style=social"/>
    </a>
</p>
<p class="aligncenter">
    <a href="https://twitter.com/alekseykorshuk" target="_blank"> 
        <img src="https://img.shields.io/twitter/follow/alekseykorshuk?style=social"/>
    </a>
</p>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    "[Optimum Transformers](https://github.com/AlekseyKorshuk/optimum-transformers) - Accelerated NLP pipelines for fast inference on CPU and GPU. Built with Transformers, Optimum and ONNX Runtime.")
# st.sidebar.header("Settings:")
pipeline_task = st.selectbox(
    'Choose pipeline',
    SUPPORTED_TASKS.keys())

pipeline_model = st.text_input('Model name, leave blank if default:', '')
num_tests = st.number_input("Number of tests:",
                            min_value=1,
                            step=1,
                            value=100,
                            help="The number of calls will be done",
                            )

# caption = (
#     "In [HuggingArtists](https://github.com/AlekseyKorshuk/huggingartist), we can generate lyrics by a specific artist. This was made by fine-tuning a pre-trained [HuggingFace Transformer](https://huggingface.co) on parsed datasets from [Genius](https://genius.com)."
# )
#
# st.markdown(caption)

model_html = """

<div class="inline-flex flex-col" style="line-height: 1.5;">
    <div class="flex">
        <div
\t\t\tstyle="display:DISPLAY_1; margin-left: auto; margin-right: auto; width: 92px; height:92px; border-radius: 50%; background-size: cover; background-image: url(&#39;USER_PROFILE&#39;)">
        </div>
    </div>
    <div style="text-align: center; margin-top: 3px; font-size: 16px; font-weight: 800">🤖 HuggingArtists Model 🤖</div>
    <div style="text-align: center; font-size: 16px; font-weight: 800">USER_NAME</div>
    <a href="https://genius.com/artists/USER_HANDLE">
    \t<div style="text-align: center; font-size: 14px;">@USER_HANDLE</div>
    </a>
</div>
"""

if st.button("Run"):
    model_name = pipeline_model if pipeline_model is not '' else None
    from optimum.onnxruntime import ORTConfig

    ort_config = ORTConfig(quantization_approach="dynamic", extra_options={"inter_op_num_threads": 1})
    benchmark = Benchmark(pipeline_task, model_name, ort_config)
    with st.spinner(text=f"Benchmarking... This may take some time to load models..."):
        results = benchmark(num_tests, plot=False)

    st.pyplot(
        plot_benchmark(results, pipeline_task, model_name)
    )

    st.subheader("Please star this repository and follow my Twitter:")
    st.markdown(
        """
    <style>
    .aligncenter {
        text-align: center;
    }
    </style>
    <p class="aligncenter">
        <a href="https://github.com/AlekseyKorshuk/optimum-transformers" target="_blank">
            <img src="https://img.shields.io/github/stars/AlekseyKorshuk/optimum-transformers?style=social"/>
        </a>
    </p>
    <p class="aligncenter">
        <a href="https://twitter.com/alekseykorshuk" target="_blank">
            <img src="https://img.shields.io/twitter/follow/alekseykorshuk?style=social"/>
        </a>
    </p>
        """,
        unsafe_allow_html=True,
    )
