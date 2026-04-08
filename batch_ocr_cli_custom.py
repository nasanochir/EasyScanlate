#!/usr/bin/env python3
"""
Modified batch_ocr_cli.py that accepts custom translation prompts.
Reads MANHWA_CUSTOM_PROMPT environment variable for custom instructions.
"""
import os
import sys

# Import the original batch_ocr_cli
sys.path.insert(0, os.path.dirname(__file__))
from batch_ocr_cli import *

# Override translate_batch to use custom prompt
def translate_batch_custom(model, api_key, api_type='gemini', target_lang='English', model_name=None):
    """Translate all OCR results with custom prompt support."""
    if not model.ocr_results:
        return
    
    # Get custom prompt from environment
    custom_prompt_prefix = os.environ.get('MANHWA_CUSTOM_PROMPT', '')
    
    # Filter watermarks and sound effects (same as original)
    watermark_keywords = ['manhwa18', 'manhwa18.com', 'read or download']
    filtered_results = []
    removed_count = 0
    
    def is_korean_char(c):
        return '\uac00' <= c <= '\ud7a3' or '\u1100' <= c <= '\u11ff' or '\u3130' <= c <= '\u318f'
    
    for result in model.ocr_results:
        text = result.get('text', '').lower().strip()
        
        if any(keyword in text for keyword in watermark_keywords):
            result['is_deleted'] = True
            removed_count += 1
            print(f"  [FILTERED - Watermark] '{result.get('text', '')}'")
            continue
        
        original_text = result.get('text', '').strip()
        korean_count = sum(is_korean_char(c) for c in original_text)
        
        if korean_count > 0:
            filtered_results.append(result)
            continue
        
        if len(original_text) <= 6:
            alpha_count = sum(c.isalpha() for c in original_text)
            symbol_count = sum(c in '!@#$%^&*()_+-=[]{}|;:,.<>?/~`' for c in original_text)
            
            if alpha_count >= 2:
                result['is_deleted'] = True
                removed_count += 1
                print(f"  [FILTERED - Sound effect] '{original_text}'")
                continue
            
            if symbol_count > len(original_text) * 0.4:
                result['is_deleted'] = True
                removed_count += 1
                print(f"  [FILTERED - Sound effect] '{original_text}'")
                continue
        
        filtered_results.append(result)
    
    if removed_count > 0:
        print(f"\nFiltered out {removed_count} watermarks and sound effects")
    
    if not filtered_results:
        print("\nNo text to translate after filtering")
        return
        
    print(f"\nTranslating {len(filtered_results)} text blocks with {api_type.upper()}...")
    
    try:
        if api_type == 'gemini':
            import google.generativeai as genai
            genai.configure(api_key=api_key)
            model_name = model_name or 'gemini-2.5-flash'
            model_api = genai.GenerativeModel(model_name)
            print(f"Using model: {model_name}")
        elif api_type == 'openai':
            from openai import OpenAI
            client = OpenAI(api_key=api_key)
            model_name = model_name or 'gpt-4.1'
            print(f"Using model: {model_name}")
        else:
            from mistralai import Mistral
            client = Mistral(api_key=api_key)
            model_name = model_name or 'mistral-small-latest'
            print(f"Using model: {model_name}")
        
        if custom_prompt_prefix:
            print(f"  Using custom translation prompt")
        
        # Batch translate in chunks of 50
        chunk_size = 50
        for i in range(0, len(filtered_results), chunk_size):
            chunk = filtered_results[i:i+chunk_size]
            texts = [r['text'] for r in chunk]
            
            combined = '\n'.join([f"{j+1}. {text}" for j, text in enumerate(texts)])
            
            # Build prompt with custom prefix if provided
            if custom_prompt_prefix:
                prompt = f"{custom_prompt_prefix}\n\nTranslate these {len(texts)} items to {target_lang}. Keep exact numbering:\n\n{combined}"
            else:
                prompt = f"Translate the following Korean text to {target_lang}. You must translate ALL {len(texts)} items and keep the exact numbering (1., 2., 3., etc.). Even single letters or short exclamations must be translated:\n\n{combined}"
            
            if api_type == 'gemini':
                response = model_api.generate_content(prompt)
                result_text = response.text.strip()
            elif api_type == 'openai':
                response = client.chat.completions.create(
                    model=model_name,
                    messages=[
                        {"role": "system", "content": f"You are a translator. Translate to {target_lang}. Keep the numbering format."},
                        {"role": "user", "content": prompt}
                    ]
                )
                result_text = response.choices[0].message.content.strip()
            else:
                response = client.chat.complete(
                    model=model_name,
                    messages=[{"role": "user", "content": prompt}]
                )
                result_text = response.choices[0].message.content.strip()
            
            translations = []
            for line in result_text.split('\n'):
                line = line.strip()
                if '. ' in line:
                    parts = line.split('. ', 1)
                    if len(parts) == 2:
                        translations.append(parts[1])
                elif line and not line[0].isdigit():
                    translations.append(line)
            
            while len(translations) < len(chunk):
                missing_idx = len(translations)
                translations.append(chunk[missing_idx]['text'])
                print(f"  Warning: Missing translation for item {i + missing_idx + 1}, using original")
            
            for j, (result, translation) in enumerate(zip(chunk, translations[:len(chunk)])):
                if 'translations' not in result:
                    result['translations'] = {}
                result['translations'][target_lang] = translation
            
            print(f"  Translated {min(i+chunk_size, len(filtered_results))}/{len(filtered_results)}")
        
        if target_lang not in model.profiles:
            model.profiles[target_lang] = {}
        model.active_profile_name = target_lang
        
        print(f"Translation complete! Active profile set to: {target_lang}")
        
    except Exception as e:
        print(f"Translation error: {e}")
        import traceback
        traceback.print_exc()


# Replace the original translate_batch with our custom version
translate_batch = translate_batch_custom

if __name__ == '__main__':
    sys.exit(main())
