"""
Customer Escalation Risk Classifier - Web Application
Real-time prediction of high-risk customer complaints
"""

import streamlit as st
import pickle
import pandas as pd
import numpy as np

# Page configuration
st.set_page_config(
    page_title="Escalation Risk Classifier",
    page_icon="🚨",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .big-font {
        font-size:20px !important;
        font-weight: bold;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
    }
    </style>
    """, unsafe_allow_html=True)

# Load model bundle
@st.cache_resource
def load_model():
    """Load the trained model and vectorizer"""
    try:
        with open("escalation_model_bundle.pkl", "rb") as f:
            bundle = pickle.load(f)
        return bundle["model"], bundle["vectorizer"], bundle["threshold"]
    except FileNotFoundError:
        st.error("⚠️ Model file not found. Please ensure 'escalation_model_bundle.pkl' is in the same directory.")
        st.stop()
    except Exception as e:
        st.error(f"⚠️ Error loading model: {str(e)}")
        st.stop()

model, vectorizer, threshold = load_model()

# Sidebar
with st.sidebar:
    st.header("📊 About This System")
    st.info(
        """
        This ML system analyzes customer complaints to identify 
        high-risk cases requiring immediate attention.
        
        **Performance Metrics:**
        - 87% recall for high-risk cases
        - 70% precision
        - Optimized threshold: 0.41
        
        **Use Cases:**
        - Customer support triage
        - Proactive escalation prevention
        - Resource allocation optimization
        """
    )
    
    st.markdown("---")
    
    st.header("🎯 Risk Levels")
    st.error("🔴 **HIGH RISK** (≥41%): Immediate attention required")
    st.warning("🟡 **MEDIUM RISK** (20-40%): Monitor closely")
    st.success("🟢 **LOW RISK** (<20%): Standard processing")
    
    st.markdown("---")
    
    st.markdown("""
        **Built with:**
        - Logistic Regression
        - TF-IDF Vectorization
        - Scikit-learn
        - Streamlit
    """)

# Main title
st.title("🚨 Customer Escalation Risk Classifier")
st.markdown("### Predict escalation risk from customer complaints in real-time")
st.markdown("---")

# Create tabs
tab1, tab2, tab3 = st.tabs(["🔍 Single Prediction", "📁 Batch Analysis", "📈 Model Info"])

with tab1:
    st.subheader("Enter a customer complaint to analyze")
    
    # Text input
    complaint_text = st.text_area(
        "Customer Complaint:",
        height=200,
        placeholder="Example: This company committed fraud and refuses to refund my money. I have contacted them multiple times with no response. I will be filing a lawsuit and reporting them to the CFPB...",
        help="Enter the full text of the customer complaint"
    )
    
    # Analyze button
    col1, col2, col3 = st.columns([1, 1, 3])
    with col1:
        analyze_button = st.button("🔍 Analyze Risk", type="primary", use_container_width=True)
    with col2:
        clear_button = st.button("🗑️ Clear", use_container_width=True)
    
    if clear_button:
        st.rerun()
    
    if analyze_button:
        if complaint_text.strip():
            with st.spinner("Analyzing complaint..."):
                # Vectorize text
                vec = vectorizer.transform([complaint_text])
                
                # Predict probability
                prob = model.predict_proba(vec)[0][1]
                
                # Apply threshold
                is_high_risk = prob >= threshold
                is_medium_risk = (prob >= 0.20) and (prob < threshold)
                
                # Display results
                st.markdown("---")
                st.markdown("## 📊 Analysis Results")
                
                # Risk level display
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    if is_high_risk:
                        st.error("### ⚠️ HIGH RISK")
                        st.markdown("**Action Required:**")
                        st.markdown("- Route to senior agent")
                        st.markdown("- Respond within 2 hours")
                        st.markdown("- Flag for management review")
                    elif is_medium_risk:
                        st.warning("### 🟡 MEDIUM RISK")
                        st.markdown("**Recommended Actions:**")
                        st.markdown("- Monitor closely")
                        st.markdown("- Prioritize in queue")
                        st.markdown("- Prepare escalation path")
                    else:
                        st.success("### ✅ LOW RISK")
                        st.markdown("**Recommended Actions:**")
                        st.markdown("- Standard processing")
                        st.markdown("- Normal queue priority")
                        st.markdown("- Routine follow-up")
                
                with col2:
                    st.metric(
                        "Escalation Probability",
                        f"{prob:.1%}",
                        delta=f"{(prob - threshold):.1%} vs threshold" if prob >= threshold else f"{(prob - threshold):.1%} vs threshold"
                    )
                    
                    # Risk gauge
                    st.progress(min(prob, 1.0))
                    
                with col3:
                    # Calculate confidence
                    if prob >= threshold:
                        confidence = prob
                    else:
                        confidence = 1 - prob
                    
                    st.metric("Model Confidence", f"{confidence:.1%}")
                    
                    if confidence > 0.8:
                        st.success("High confidence prediction")
                    elif confidence > 0.6:
                        st.info("Moderate confidence")
                    else:
                        st.warning("Low confidence - manual review recommended")
                
                # Risk factors analysis
                st.markdown("---")
                st.markdown("## 🔍 Detected Risk Factors")
                
                factors_found = []
                complaint_lower = complaint_text.lower()
                
                # Check for various risk indicators
                if any(word in complaint_lower for word in ['fraud', 'scam', 'theft', 'steal', 'stolen']):
                    factors_found.append(("🔴 HIGH", "Fraud/theft language detected"))
                
                if any(word in complaint_lower for word in ['lawyer', 'attorney', 'sue', 'lawsuit', 'legal action', 'court']):
                    factors_found.append(("🔴 HIGH", "Legal threat indicators"))
                
                if any(word in complaint_lower for word in ['cfpb', 'ftc', 'bbb', 'attorney general', 'regulatory']):
                    factors_found.append(("🔴 HIGH", "Regulatory complaint threat"))
                
                if any(word in complaint_lower for word in ['urgent', 'immediately', 'asap', 'emergency', 'critical']):
                    factors_found.append(("🟡 MEDIUM", "Urgency keywords present"))
                
                if complaint_text.count('!') > 2:
                    factors_found.append(("🟡 MEDIUM", f"High exclamation usage ({complaint_text.count('!')} marks)"))
                
                caps_ratio = sum(c.isupper() for c in complaint_text) / len(complaint_text) if len(complaint_text) > 0 else 0
                if caps_ratio > 0.15:
                    factors_found.append(("🟡 MEDIUM", f"Excessive capitalization ({caps_ratio:.0%})"))
                
                if any(word in complaint_lower for word in ['angry', 'furious', 'outraged', 'disgusted', 'horrible']):
                    factors_found.append(("🟡 MEDIUM", "Strong negative emotion language"))
                
                if any(word in complaint_lower for word in ['refund', 'money back', 'reimburse']):
                    factors_found.append(("🟢 LOW", "Financial restitution mentioned"))
                
                # Display factors
                if factors_found:
                    factor_df = pd.DataFrame(factors_found, columns=["Severity", "Factor"])
                    
                    for _, row in factor_df.iterrows():
                        if "HIGH" in row["Severity"]:
                            st.error(f"{row['Severity']}: {row['Factor']}")
                        elif "MEDIUM" in row["Severity"]:
                            st.warning(f"{row['Severity']}: {row['Factor']}")
                        else:
                            st.info(f"{row['Severity']}: {row['Factor']}")
                else:
                    st.success("✅ No significant risk indicators detected")
                
                # Text statistics
                st.markdown("---")
                st.markdown("## 📝 Complaint Statistics")
                
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Character Count", len(complaint_text))
                col2.metric("Word Count", len(complaint_text.split()))
                col3.metric("Exclamation Marks", complaint_text.count('!'))
                col4.metric("Question Marks", complaint_text.count('?'))
                
        else:
            st.warning("⚠️ Please enter a complaint to analyze")

with tab2:
    st.subheader("Batch Analysis")
    st.markdown("Upload a CSV file with customer complaints for batch processing")
    
    uploaded_file = st.file_uploader(
        "Choose a CSV file",
        type=['csv'],
        help="CSV should have a column named 'complaint' or 'text' with complaint narratives"
    )
    
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            
            st.success(f"✅ File loaded successfully! {len(df)} complaints found.")
            
            # Find the text column
            text_column = None
            for col in ['complaint', 'text', 'narrative', 'Consumer complaint narrative']:
                if col in df.columns:
                    text_column = col
                    break
            
            if text_column:
                if st.button("🚀 Analyze All Complaints"):
                    with st.spinner(f"Analyzing {len(df)} complaints..."):
                        # Vectorize
                        X = vectorizer.transform(df[text_column].fillna(''))
                        
                        # Predict
                        probabilities = model.predict_proba(X)[:, 1]
                        predictions = (probabilities >= threshold).astype(int)
                        
                        # Add results to dataframe
                        df['escalation_probability'] = probabilities
                        df['risk_level'] = predictions
                        df['risk_category'] = df['escalation_probability'].apply(
                            lambda x: 'HIGH' if x >= threshold else ('MEDIUM' if x >= 0.20 else 'LOW')
                        )
                        
                        # Display summary
                        st.markdown("---")
                        st.markdown("## 📊 Batch Analysis Summary")
                        
                        col1, col2, col3, col4 = st.columns(4)
                        
                        high_risk_count = (df['risk_category'] == 'HIGH').sum()
                        medium_risk_count = (df['risk_category'] == 'MEDIUM').sum()
                        low_risk_count = (df['risk_category'] == 'LOW').sum()
                        avg_prob = df['escalation_probability'].mean()
                        
                        col1.metric("Total Complaints", len(df))
                        col2.metric("🔴 High Risk", high_risk_count, f"{high_risk_count/len(df)*100:.1f}%")
                        col3.metric("🟡 Medium Risk", medium_risk_count, f"{medium_risk_count/len(df)*100:.1f}%")
                        col4.metric("🟢 Low Risk", low_risk_count, f"{low_risk_count/len(df)*100:.1f}%")
                        
                        # Show high-risk complaints first
                        st.markdown("### 🔴 High-Risk Complaints (Immediate Attention Required)")
                        high_risk_df = df[df['risk_category'] == 'HIGH'].sort_values('escalation_probability', ascending=False)
                        
                        if len(high_risk_df) > 0:
                            st.dataframe(
                                high_risk_df[[text_column, 'escalation_probability', 'risk_category']].head(10),
                                use_container_width=True
                            )
                        else:
                            st.success("No high-risk complaints detected!")
                        
                        # Download results
                        st.markdown("---")
                        csv = df.to_csv(index=False)
                        st.download_button(
                            label="📥 Download Results (CSV)",
                            data=csv,
                            file_name="escalation_analysis_results.csv",
                            mime="text/csv",
                        )
            else:
                st.error("⚠️ Could not find a complaint text column. Please ensure your CSV has a column named 'complaint', 'text', or 'narrative'")
                
        except Exception as e:
            st.error(f"⚠️ Error processing file: {str(e)}")

with tab3:
    st.subheader("Model Information")
    
    st.markdown("""
    ### 🤖 Model Architecture
    
    **Algorithm:** Logistic Regression with L2 regularization
    
    **Text Processing:** TF-IDF Vectorization
    - Max features: 5000
    - N-gram range: (1, 2)
    - Min document frequency: 5
    
    **Class Handling:** Balanced class weights to handle imbalance
    
    ### 📊 Performance Metrics
    
    Evaluated on test set of 1,097 complaints:
    
    | Metric | Low Risk | High Risk |
    |--------|----------|-----------|
    | Precision | 0.89 | 0.70 |
    | Recall | 0.93 | 0.87 |
    | F1-Score | 0.91 | 0.78 |
    
    **Overall Accuracy:** 87%
    
    ### 🎯 Threshold Optimization
    
    - **Default threshold:** 0.50 → 76% recall
    - **Optimized threshold:** 0.41 → 87% recall ⭐
    
    The threshold was optimized to prioritize catching high-risk cases (minimize false negatives) 
    at the cost of some false positives. This reflects the business priority of preventing 
    escalations over minimizing workload.
    
    ### 🏷️ Labeling Strategy
    
    Complaints were labeled as "High Risk" based on:
    1. **Untimely company responses** (delayed resolution)
    2. **Escalation keywords**: fraud, lawsuit, attorney, regulatory complaint, etc.
    3. **Balanced sampling**: 67% low-risk, 33% high-risk
    
    This hybrid approach creates realistic labels that reflect actual escalation patterns.
    
    ### 💡 Key Features
    
    The model identifies high-risk complaints based on:
    - Legal terminology (attorney, lawsuit, sue)
    - Fraud/crime language (fraud, scam, theft)
    - Regulatory threats (CFPB, FTC, BBB)
    - Urgency indicators (immediate, urgent, ASAP)
    - Emotion intensity (anger, outrage, excessive caps/punctuation)
    
    ### 🚀 Production Considerations
    
    **Inference Speed:** ~2ms per complaint
    
    **Deployment:** Lightweight model (< 50MB) suitable for real-time API
    
    **Interpretability:** Linear model allows inspection of feature weights
    
    **Monitoring:** Track precision-recall metrics in production to detect drift
    """)

# Example complaints section (always visible at bottom)
st.markdown("---")
st.markdown("## 💡 Try These Example Complaints")

col1, col2 = st.columns(2)

with col1:
    st.markdown("### 🔴 High-Risk Example")
    high_risk_example = """This company is engaging in FRAUDULENT practices! I have been trying to resolve this billing error for THREE MONTHS with absolutely NO response from your team. The unauthorized charges continue to appear on my account despite my repeated complaints.

I am contacting my attorney tomorrow and will be filing a formal complaint with the CFPB and FTC. I will also be pursuing a lawsuit for the stress and financial damage this has caused. This is completely UNACCEPTABLE and ILLEGAL business practice!

I DEMAND an immediate refund of all charges and compensation for my time. If I don't hear back within 24 hours, I will proceed with legal action."""
    
    st.text_area("Example complaint:", value=high_risk_example, height=250, key="high_ex", disabled=True)
    
    if st.button("📋 Copy to Analyzer", key="copy_high"):
        st.session_state['complaint_to_analyze'] = high_risk_example
        st.success("✅ Example copied! Go to 'Single Prediction' tab to analyze.")

with col2:
    st.markdown("### 🟢 Low-Risk Example")
    low_risk_example = """Hello,

I recently opened a checking account with your bank and have a question about the monthly maintenance fee structure. According to the terms I received, there should be no fee if I maintain a minimum balance of $1,500.

However, I noticed a $12 fee on my first statement. My balance has remained above $2,000 since opening the account. Could you please clarify why this fee was charged and help me understand how to avoid it going forward?

Thank you for your assistance. I appreciate your help with this matter.

Best regards"""
    
    st.text_area("Example complaint:", value=low_risk_example, height=250, key="low_ex", disabled=True)
    
    if st.button("📋 Copy to Analyzer", key="copy_low"):
        st.session_state['complaint_to_analyze'] = low_risk_example
        st.success("✅ Example copied! Go to 'Single Prediction' tab to analyze.")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; padding: 20px;'>
    <p>Built with ❤️ using Streamlit | Customer Escalation Risk Classifier v1.0</p>
</div>
""", unsafe_allow_html=True)
