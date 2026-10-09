from backend.llm.stream_parser import parse_answer_stream

stream = [
    '{\n', 
    '  "ans', 
    'wer": "Hello ', 
    'World!\\nHow ', 
    'are you?\\"",\n  "claims": []\n}'
]
parser = parse_answer_stream(stream)
for event, data in parser:
    print(event, repr(data))
