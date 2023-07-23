import streamlit as st
from st_files_connection import FilesConnection

st.set_page_config(
    page_title="Home",
    page_icon="👋",
)
# --- GENERAL SETTINGS ---
STRIPE_CHECKOUT = "https://buy.stripe.com/test_fZe7tp7St2Hm3QI9AA"
CONTACT_EMAIL = "YOUREMAIL@EMAIL.COM"
DEMO_VIDEO = ""
PRODUCT_NAME = "PubMeta.ai: Research Tool"
PRODUCT_TAGLINE = "Ready To find the right treament for your Condition? 🚀"
PRODUCT_DESCRIPTION = """
PubMeta.ai is the ultimate medical research companion that empowers you to make informed health decisions. It combines the latest cutting-edge medical research with user reports, all in one place. With PubMeta.ai, you'll enjoy the following powerful features:

- Integration with PubMed: Access the latest scientific research and studies directly from PubMed.
- User-Reported Data: Gain valuable insights from user-reported data, providing real-world perspectives and experiences.
- Side-by-Side Treatment Comparison: Compare medical treatments effortlessly, evaluating efficacy, side effects, costs, and patient feedback.
- Personalized Recommendations: Receive tailored insights and treatment recommendations based on your specific preferences and health concerns.
- Historical Research Analysis: Dive into historical research and studies, combining them with the latest cutting-edge knowledge for a comprehensive understanding.
- Real-World Perspectives: Explore anonymized health data from various online communities, gaining broad insights into medical trends and experiences.
- Holistic Approach: PubMeta.ai creates a patient-oriented healthcare ecosystem, considering both scientific research and real-life patient experiences.
- Constant Updates: Stay up-to-date with regular platform updates, ensuring you have access to the latest advancements in medical knowledge.
- Transparent and Accessible: PubMeta.ai makes healthcare data transparent and easily accessible, empowering you to make informed health decisions.
- Dedicated Support: Our dedicated support team is ready to assist you with any questions or concerns along your healthcare journey.

**PubMeta.ai is your essential tool for navigating the world of medical research with confidence. Unlock the power of knowledge and transform your healthcare decisions today!**
"""


st.header(PRODUCT_NAME)

st.markdown(
    """
# Welcome to PubMeta.ai! 👋

PubMeta.ai is an innovative medical chatbot and research tool inspired by medical resources like PubMed and WebMD. Our mission is to create a holistic, evidence-based, patient-oriented healthcare ecosystem.

By integrating artificial intelligence, natural language processing, and user-reported data, PubMeta.ai provides reliable, personalized, and up-to-date medical advice. 

## Unique Features
Our chatbot cross-references the user-reported data with the latest studies from [PubMed](https://pubmed.ncbi.nlm.nih.gov/), ensuring the advice provided is backed by recent scientific findings. 

Simultaneously, PubMeta.ai aggregates and analyzes anonymized health data from various online communities, providing broad perspectives on real-world medical trends and insights.

**👈 Explore the functionalities in the sidebar** to experience the revolutionary approach of PubMeta.ai!

## Compare Medical Treatments Like Never Before
PubMeta.ai offers a groundbreaking feature enabling users to compare medical treatments similar to how one compares cars in a marketplace. It assesses treatment options based on efficacy, side effects, costs, and real-world patient feedback. 

### Want to learn more?
- Visit our website [pubmeta.ai](https://pubmeta.ai)
- Dive into our [documentation](https://docs.pubmeta.ai)
- Engage in our [community forums](https://discuss.pubmeta.ai)

### Experience the Transformation 
At PubMeta.ai, we aim to revolutionize how individuals navigate their health journeys by making healthcare data as transparent and accessible as possible.

"""
)

st.markdown("""END USER LICENSE AGREEMENT (EULA)

IMPORTANT: PLEASE READ THIS END USER LICENSE AGREEMENT CAREFULLY BEFORE USING OUR MEDICAL RESEARCH AI TOOL. BY USING THE TOOL, YOU ACKNOWLEDGE THAT YOU HAVE READ AND AGREE TO BE BOUND BY THE TERMS AND CONDITIONS OF THIS AGREEMENT.

1. DEFINITIONS
   1.1. "Tool" refers to the medical research AI tool provided on our website.
   1.2. "We," "us," or "our" refers to [Your Company].
   1.3. "You" or "your" refers to the end user of the Tool.

2. LICENSE GRANT
   2.1. We grant you a non-exclusive, non-transferable, revocable license to use the Tool solely for your personal and non-commercial purposes, subject to the terms and conditions of this EULA.
   2.2. You may not modify, adapt, translate, reverse engineer, decompile, disassemble, or create derivative works of the Tool.

3. INTELLECTUAL PROPERTY RIGHTS
   3.1. The Tool and all associated intellectual property rights belong to us or our licensors. The Tool is protected by copyright and other intellectual property laws.
   3.2. You agree not to remove, alter, or obscure any copyright, trademark, or other proprietary notices on the Tool.

4. DISCLAIMER OF WARRANTIES
   4.1. The Tool is provided on an "as is" basis, without any warranties or representations of any kind, whether express or implied.
   4.2. We do not warrant that the Tool will be error-free, accurate, reliable, or suitable for your intended use.
   4.3. You acknowledge that the Tool is not intended to provide medical advice, diagnosis, or treatment, and should not replace professional medical judgment or consultation.

5. LIMITATION OF LIABILITY
   5.1. To the maximum extent permitted by law, we shall not be liable for any direct, indirect, incidental, consequential, or special damages arising out of or in connection with the use of the Tool.
   5.2. You agree to indemnify and hold us harmless from any claims, damages, liabilities, or expenses arising out of your use of the Tool.

6. GOVERNING LAW AND DISPUTE RESOLUTION
   6.1. This EULA shall be governed by and construed in accordance with the laws of [Your Jurisdiction].
   6.2. Any dispute arising out of or in connection with this EULA shall be subject to the exclusive jurisdiction of the courts of [Your Jurisdiction].

7. GENERAL PROVISIONS
   7.1. This EULA constitutes the entire agreement between you and us regarding the subject matter herein and supersedes all prior or contemporaneous agreements.
   7.2. If any provision of this EULA is found to be invalid or unenforceable, the remaining provisions shall remain in full force and effect.
   7.3. Our failure to enforce any right or provision of this EULA shall not constitute a waiver of such right or provision.

By using our medical research AI tool, you agree to be bound by the terms and conditions of this EULA.

""")