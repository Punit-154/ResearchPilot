import json

def parse_answer_stream(token_stream):
    """
    Consumes a stream of LLM tokens.
    Extracts and yields ('answer_chunk', text) live as the "answer" field is generated.
    Finally yields ('full_json', full_raw_string).
    """
    full_text = ""
    in_answer = False
    answer_started = False
    escape_next = False
    
    for chunk in token_stream:
        full_text += chunk
        
        if not answer_started:
            # Look for "answer": "
            idx = full_text.find('"answer"')
            if idx != -1:
                colon_idx = full_text.find(':', idx)
                if colon_idx != -1:
                    quote_idx = full_text.find('"', colon_idx)
                    if quote_idx != -1:
                        answer_started = True
                        in_answer = True
                        # Everything after this quote is the start of the answer
                        ans_part = full_text[quote_idx + 1:]
                        
                        # Process this initial part
                        clean_part = ""
                        for char in ans_part:
                            if escape_next:
                                clean_part += char
                                escape_next = False
                            elif char == '\\':
                                escape_next = True
                            elif char == '"':
                                in_answer = False
                                break
                            else:
                                clean_part += char
                        
                        if clean_part:
                            yield ('answer_chunk', clean_part)
        elif in_answer:
            clean_part = ""
            for char in chunk:
                if escape_next:
                    if char == 'n': clean_part += '\n'
                    elif char == 'r': clean_part += '\r'
                    elif char == 't': clean_part += '\t'
                    else: clean_part += char
                    escape_next = False
                elif char == '\\':
                    escape_next = True
                elif char == '"':
                    in_answer = False
                    break
                else:
                    clean_part += char
            
            if clean_part:
                yield ('answer_chunk', clean_part)
                
    yield ('full_json', full_text)
