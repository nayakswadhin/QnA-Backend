from transformers import AutoModelForQuestionAnswering, AutoTokenizer, pipeline

model_name = "deepset/roberta-base-squad2"

context = 'The option to convert models between FARM and transformers gives freedom to the user and let people easily switch between frameworks.'
question = 'Why is model conversion important?'

nlp = pipeline('question-answering', model=model_name, tokenizer=model_name)

# a) Get predictions
def nlp_qna(context, question):
    QA_input = {
        'context' : context,
        'question': question
    }
    res = nlp(QA_input)
    return res
