with tab1:
    st.subheader("Upload your CV or paste text")
    
    uploaded_file = st.file_uploader("Upload CV (txt/pdf)", type=["txt", "pdf"])
    user_input = st.text_area("Or paste your CV text here:")

    # Function to extract text from PDF or TXT
    import PyPDF2
    def extract_text(file):
        if file.type == "application/pdf":
            pdf_reader = PyPDF2.PdfReader(file)
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text() + "\n"
            return text
        else:  # assume plain text
            return file.read().decode("utf-8")

    if st.button("Analyze CV"):
        if uploaded_file is not None:
            text = extract_text(uploaded_file)
        else:
            text = user_input

        if not text.strip():
            st.warning("Please provide some CV text.")
        else:
            st.success("CV text loaded successfully!")

            # --- NLP Analysis ---
            st.subheader("NLP Analysis")
            results, similarities, vectorizer = nlp.find_top_categories(
                text, JSON_FILE, top_n=5
            )
            st.write("**Top categories (NLP):**")
            for category, score in results:
                st.write(f"- {category} ({score:.2%})")

            best_category = results[0][0]
            category_data = nlp.load_category_data(JSON_FILE)
            missing = nlp.find_missing_keywords(
                text, best_category, category_data, vectorizer
            )
            st.write(f"**Missing skills for {best_category}:**")
            if missing:
                for word, importance, _, _ in missing:
                    st.write(f"- {word} (importance: {importance:.2f})")
            else:
                st.success("No major skill gaps found!")

            # --- Deep Learning Prediction ---
            st.subheader("Deep Learning Model (BERT)")
            model, label_encoder, tokenizer = DeepLearningModel.load_trained_model(DL_MODEL_PATH)
            pred_cat, confidence, top_preds = DeepLearningModel.predict_cv_category(
                text, model, label_encoder, tokenizer
            )
            st.write(f"**Predicted Career Path:** {pred_cat} ({confidence*100:.2f}%)")
            st.write("**Top predictions:**")
            for cat, conf in top_preds:
                st.write(f"- {cat}: {conf*100:.2f}%")
