from collections import defaultdict

prompt_dict = {
    "sst2":{
        "single_example_prompt": (
            "##Review: {sentence}\n"
            "##Positive or negative: {label_text}\n"
        ),
        "question_prompt": (
            '##Review: {sentence}\n'
        ),
        "multi_example_prompt_no_reasoning": (
            "<s>[INST] <<SYS>>"
            "You are a psychologist. You are precise. You analyze sentiment of people's reviews."
            "<</SYS>>"
            "Given the examples, classify the review with "
            "a single word \'positive\' or \'negative\'.\n"
            "Here are the examples:\n"
            "{examples}"
            "\nHere is the review which you need to classify with a single word \'positive\' or \'negative\':\n{question}"
            "##Positive or negative: This review is"
            "[/INST]"
        ),
        "multi_example_prompt": (
            "<s>[INST] <<SYS>>"
            "You are a psychologist. You are precise. You analyze sentiment of people's reviews."
            "<</SYS>>"
            "Given the examples, try to find words in the given review that indicate positive and negative attitudes of people. Then analyze which side is stronger.\n"
            "Here are the examples:\n"
            "{examples}"
            "\nHere is the review which you need to find words in that indicate positive and negative attitudes of people and then analyze which side is stronger.:\n{question}"
            "[/INST]"
        ),
        "single_get_answer_prompt": (
            "##Review: {sentence}\n"
            "##Positive or negative: {label_text}\n"
        ),
        "multi_get_answer_prompt": (
            "<s>[INST] <<SYS>>"
            "You are a psychologist. You are precise. You analyze sentiment of people's reviews. You respect the analyzation."
            "<</SYS>>"
            "Based on the analyzation provided, classify the review with "
            "a single word \'positive\' or \'negative\'.\n"
            "\nHere is the review you need to classify with a single word \'positive\' or \'negative\':\n{question}##Analyzation: {reasoning}\n"
            "##Positive or negative: "
            "[/INST]"
        ), 
        "label_map":{
            'positive': 1,
            'negative': 0
        },
        "label_map_reverse":{
            1: 'positive',
            0: 'negative'
        }
    }, 
    "gsm8k": {
        "single_example_prompt": (
            "##Problem: {question}\n"
            "##Reasoning: {reasoning}\n"
        ),
        "question_prompt": (
            "##Problem: {question}\n"
        ),
        "multi_example_prompt": (
            "<s>[INST] <<SYS>>"
            "You are a student in a math exam. You are precise. You do not greet people."
            "<</SYS>>"
            "Given the examples, give a short reasoning within 100 words for the problem.\n"
            "Here are the examples:\n"
            "{examples}"
            "\nHere is the question you need to reason about:\n{question}##Reasoning: "
            "[/INST]"
        ),
        "single_get_answer_prompt": (
            "##Problem: {question}\n"
            "##Reasoning: {reasoning}\n"
            "##Answer: {correct}\n"
        ),
        "multi_get_answer_prompt": (
            "<s>[INST] <<SYS>>"
            "You are a student in a math exam. You are precise. You do not greet people."
            "<</SYS>>"
            "Based on the reasoning provided, provide the correct answer "
            "by writing a single number.\n"
            "Here are the examples:\n"
            "{examples}"
            "\nHere is the question you need to answer with a single number:\n{question}##Reasoning: {reasoning}\n"
            "##Answer: "
            "[/INST]"
        )
    }
}