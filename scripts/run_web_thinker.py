# run_web_thinker.py
import os
import json
import time
import re
from tqdm import tqdm
import numpy as np
import torch
import string
from typing import Optional, Tuple, List, Dict, Set
import argparse
import random
import asyncio
import aiohttp

from openai import AsyncOpenAI

from search.bing_search import (
    bing_web_search, 
    extract_relevant_info, 
    fetch_page_content, 
    fetch_page_content_async,
    extract_snippet_with_context,
    bing_web_search_async
)
from evaluate.evaluate import (
    run_evaluation, 
    extract_answer_fn
)
from prompts.prompts import (
    get_gpqa_search_o1_instruction, 
    get_gpqa_web_thinker_instruction, 
    get_deep_web_explorer_instruction, 
    get_web_page_reader_instruction,
    get_search_intent_instruction,
    get_click_intent_instruction,
    get_math_search_o1_instruction, 
    get_code_search_o1_instruction, 
    get_singleqa_search_o1_instruction, 
    get_multiqa_search_o1_instruction, 
    get_task_instruction_openqa, 
    get_task_instruction_math, 
    get_task_instruction_multi_choice, 
    get_task_instruction_code, 
)
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("YOUR_QWQ_PATH")
aux_tokenizer = AutoTokenizer.from_pretrained("YOUR_QWEN2.5_PATH")


# Define special tokens
BEGIN_SEARCH_QUERY = "<|begin_search_query|>"
END_SEARCH_QUERY = "<|end_search_query|>"
BEGIN_SEARCH_RESULT = "<|begin_search_result|>"
END_SEARCH_RESULT = "<|end_search_result|>"

BEGIN_CLICK_LINK = "<|begin_click_link|>"
END_CLICK_LINK = "<|end_click_link|>"
# BEGIN_CLICK_INTENT = "<|begin_click_intent|>"
# END_CLICK_INTENT = "<|end_click_intent|>"
BEGIN_CLICK_RESULT = "<|begin_click_result|>"
END_CLICK_RESULT = "<|end_click_result|>"

error_indicators = [
    'limit exceeded',
    'Error fetching',
    'Account balance not enough',
    'Invalid bearer token',
    'HTTP error occurred',
    'Error: Connection error occurred',
    'Error: Request timed out',
    'Unexpected error',
    'Please turn on Javascript',
    'Enable JavaScript',
    'port=443',
    'Please enable cookies',
]

def parse_args():
    parser = argparse.ArgumentParser(description="Run Search-o1 for various datasets and models.")
    parser.add_argument('--single_question', type=str, default=None, help="Single question to process instead of dataset")
    parser.add_argument('--dataset_name', type=str, required=False, default='custom', help="Name of the dataset to use.")
    parser.add_argument('--split', type=str, required=False, default='test', help="Dataset split to use.")
    parser.add_argument('--subset_num', type=int, default=-1, help="Number of examples to process. Defaults to all if not specified.")

    parser.add_argument('--temperature', type=float, default=0.7, help="Sampling temperature.")
    parser.add_argument('--top_p', type=float, default=0.8, help="Top-p sampling parameter.")
    parser.add_argument('--min_p', type=float, default=0.05, help="Minimum p sampling parameter.")
    parser.add_argument('--top_k_sampling', type=int, default=20, help="Top-k sampling parameter.")
    parser.add_argument('--repetition_penalty', type=float, default=1.05, help="Repetition penalty. If not set, defaults based on the model.")
    parser.add_argument('--max_tokens', type=int, default=32768, help="Maximum number of tokens to generate. If not set, defaults based on the model and dataset.")

    parser.add_argument('--max_search_limit', type=int, default=10, help="Maximum number of searches per question.")
    parser.add_argument('--top_k', type=int, default=10, help="Maximum number of search documents to return.")
    parser.add_argument('--keep_links', action='store_true', default=False, help="Whether to keep links in fetched web content")
    parser.add_argument('--use_jina', action='store_true', help="Whether to use Jina API for document fetching.")
    parser.add_argument('--jina_api_key', type=str, default='None', help="Your Jina API Key to Fetch URL Content.")
    parser.add_argument('--bing_subscription_key', type=str, required=True, help="Bing Search API subscription key.")
    parser.add_argument('--bing_endpoint', type=str, default="https://api.bing.microsoft.com/v7.0/search", help="Bing Search API endpoint.")
    parser.add_argument('--eval', action='store_true', help="Whether to run evaluation after generation.")
    parser.add_argument('--seed', type=int, default=None, help="Random seed for generation. If not set, will use current timestamp as seed.")
    parser.add_argument('--api_base_url', type=str, required=True, help="Base URL for the API endpoint")
    parser.add_argument('--aux_api_base_url', type=str, required=True, help="Base URL for the auxiliary model API endpoint")
    parser.add_argument('--model_name', type=str, default="QwQ-32B", help="Name of the model to use")
    parser.add_argument('--aux_model_name', type=str, default="Qwen2.5-72B-Instruct", help="Name of the auxiliary model to use")
    parser.add_argument('--concurrent_limit', type=int, default=32, help="Maximum number of concurrent API calls")
    parser.add_argument('--lora_name', type=str, default=None, help="Name of the LoRA adapter to load")
    parser.add_argument('--lora_path', type=str, default=None, help="Path to the LoRA weights")
    return parser.parse_args()



def extract_between(text, start_marker, end_marker):
    """Extracts text between two markers in a string."""
    try:
        pattern = re.escape(end_marker[::-1]) + r"(.*?)" + re.escape(start_marker[::-1])
        # Run pattern matching with timeout
        matches = re.findall(pattern, text[::-1], flags=re.DOTALL)
        if matches:
            return matches[0][::-1].strip()
        return None
    except Exception as e:
        print(f"---Error:---\n{str(e)}")
        print(f"-------------------")
        return None

def format_search_results(relevant_info: List[Dict]) -> str:
    """Format search results into a readable string"""
    formatted_documents = ""
    for i, doc_info in enumerate(relevant_info):
        doc_info['title'] = doc_info['title'].replace('<b>','').replace('</b>','')
        doc_info['snippet'] = doc_info['snippet'].replace('<b>','').replace('</b>','')
        formatted_documents += f"***Web Page {i + 1}:***\n"
        formatted_documents += json.dumps(doc_info, ensure_ascii=False, indent=2) + "\n"
        # formatted_documents += f"Title: {doc_info['title']}\n"
        # formatted_documents += f"URL: {doc_info['url']}\n"
        # formatted_documents += f"Snippet: {doc_info['snippet']}\n\n"
        # if 'page_info' in doc_info:
        #     formatted_documents += f"Web Page Information: {doc_info['page_info']}\n\n\n\n"
    return formatted_documents


async def generate_response(
    client: AsyncOpenAI,
    prompt: str,
    semaphore: asyncio.Semaphore,
    generate_mode: str = "chat",
    temperature: float = 0.0,
    top_p: float = 1.0,
    max_tokens: int = 32768,
    repetition_penalty: float = 1.0,
    top_k: int = 1,
    min_p: float = 0.0,
    model_name: str = "QwQ-32B",
    stop: List[str] = [END_SEARCH_QUERY],
    retry_limit: int = 3,
) -> Tuple[str, str]:
    """Generate a single response with retry logic"""
    for attempt in range(retry_limit):
        try:
            async with semaphore:
                if generate_mode == "chat":
                    messages = [{"role": "user", "content": prompt}]
                    if 'qwq' in model_name.lower():
                        formatted_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                    else:
                        formatted_prompt = aux_tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                else:
                    formatted_prompt = prompt

                response = await client.completions.create(
                    model=model_name,
                    prompt=formatted_prompt,
                    temperature=temperature,
                    top_p=top_p,
                    max_tokens=max_tokens,
                    stop=stop,
                    extra_body={
                        'top_k': top_k,
                        'include_stop_str_in_output': True,
                        'repetition_penalty': repetition_penalty,
                        # 'min_p': min_p
                    },
                    timeout=3600,
                )
                return formatted_prompt, response.choices[0].text
        except Exception as e:
            print(f"Generate Response Error occurred: {e}, Starting retry attempt {attempt + 1}")
            print(prompt)
            if attempt == retry_limit - 1:
                print(f"Failed after {retry_limit} attempts: {e}")
                return "", ""
            await asyncio.sleep(1 * (attempt + 1))
    return "", ""


async def generate_deep_web_explorer(
    client: AsyncOpenAI,
    aux_client: AsyncOpenAI,
    search_query: str,
    document: str,
    search_intent: str,
    args: argparse.Namespace,
    search_cache: Dict,
    url_cache: Dict,
    semaphore: asyncio.Semaphore,
) -> Tuple[str, List[Dict], str]:
    """
    Generate deep web exploration with multiple search and click operations
    Returns the output, list of interaction records, and initial prompt
    """
    prompt = get_deep_web_explorer_instruction(search_query=search_query, search_intent=search_intent, search_result=document)
    output = ""
    original_prompt = ""
    total_tokens = len(prompt.split())  # Track total tokens including prompt
    MAX_TOKENS = 20000
    MAX_INTERACTIONS = 10  # Maximum combined number of searches and clicks
    clicked_urls = set()  # Track clicked URLs
    executed_search_queries = set()  # Track executed search queries
    total_interactions = 0
    finished = False
    first_generation = True

    while True:
        # Generate next response
        formatted_prompt, response = await generate_response(
            client=client,
            prompt=prompt,
            semaphore=semaphore,
            generate_mode="chat" if first_generation else "completion",
            temperature=args.temperature,
            top_p=args.top_p,
            max_tokens=args.max_tokens,
            repetition_penalty=args.repetition_penalty,
            top_k=args.top_k_sampling,
            min_p=args.min_p,
            model_name=args.model_name,
            stop=[END_SEARCH_QUERY, END_CLICK_LINK],
        )

        if first_generation:
            original_prompt = formatted_prompt
            prompt = formatted_prompt
        
        output += response.replace('</think>\n','')
        total_tokens = len(prompt.split()) + len(response.split())
        first_generation = False

        if total_tokens >= MAX_TOKENS or total_interactions >= MAX_INTERACTIONS:
            break

        # Check for search query
        if response.rstrip().endswith(END_SEARCH_QUERY):
            new_query = extract_between(response, BEGIN_SEARCH_QUERY, END_SEARCH_QUERY)
            if new_query:
                total_interactions += 1

                if new_query in executed_search_queries:
                    # If search query was already executed, append message and continue
                    search_result = f"\n{BEGIN_SEARCH_RESULT}\nYou have already searched for this query. Please use the previously found information.\n{END_SEARCH_RESULT}\n"
                    output += search_result
                    prompt += output
                    total_tokens += len(search_result.split())
                    continue

                executed_search_queries.add(new_query)  # Add query to executed set
                
                # Execute search
                if new_query in search_cache:
                    results = search_cache[new_query]
                else:
                    try:
                        # results = bing_web_search(new_query, args.bing_subscription_key, args.bing_endpoint)
                        results = await bing_web_search_async(new_query, args.bing_subscription_key, args.bing_endpoint)
                        search_cache[new_query] = results
                    except Exception as e:
                        print(f"Error during search query '{new_query}': {e}")
                        results = {}
                print('- Searched for:', new_query)

                relevant_info = extract_relevant_info(results)[:args.top_k]

                formatted_documents = format_search_results(relevant_info)
                
                # Append search results
                search_result = f"\n{BEGIN_SEARCH_RESULT}\n{formatted_documents}\n{END_SEARCH_RESULT}\n"
                output += search_result
                prompt += output
                total_tokens += len(search_result.split())
                
        # Check for click link
        elif response.rstrip().endswith(END_CLICK_LINK):
            url = extract_between(response, BEGIN_CLICK_LINK, END_CLICK_LINK)
            # click_intent = extract_between(response, BEGIN_CLICK_INTENT, END_CLICK_INTENT)
            _, click_intent = await generate_response(
                client=aux_client,
                model_name=args.aux_model_name,
                prompt=get_click_intent_instruction(output),
                semaphore=semaphore,
            )

            if url and click_intent:
                total_interactions += 1
                if url in clicked_urls:
                    # If URL was already clicked, append message
                    click_result = f"\n{BEGIN_CLICK_RESULT}\nYou have already clicked this URL.\n{END_CLICK_RESULT}\nOK, let me use the previously found information."
                    output += click_result
                    prompt += output
                    total_tokens += len(click_result.split())
                    continue

                clicked_urls.add(url)  # Add URL to clicked set
                print(f"- Clicking on URL: {url} with intent: {click_intent}")
                # Fetch and process page content
                if url not in url_cache:
                    try:
                        content = await fetch_page_content_async(
                            [url], 
                            use_jina=args.use_jina, 
                            jina_api_key=args.jina_api_key, 
                            keep_links=args.keep_links
                        )
                        content = content[url]
                        # Only cache content if it doesn't contain error indicators
                        has_error = (any(indicator.lower() in content.lower() for indicator in error_indicators) and len(content.split()) < 64) or content == ''
                        if not has_error:
                            url_cache[url] = content
                    except Exception as e:
                        print(f"Error fetching URL {url}: {e}")
                        content = ""
                else:
                    content = url_cache[url]

                # Check if content has error indicators
                has_error = any(indicator.lower() in content.lower() for indicator in error_indicators) or content == ''
                
                if has_error:
                    # If content has error, use it directly as summary
                    summary = "Unable to fetch the page content. You can try other links."
                else:
                    # Use web page reader to summarize content
                    reader_prompt = get_web_page_reader_instruction(click_intent, content)
                    _, summary = await generate_response(
                        client=aux_client,
                        prompt=reader_prompt,
                        semaphore=semaphore,
                        max_tokens=3600,
                        model_name=args.aux_model_name,
                    )

                # Append click results
                click_result = f"\n{BEGIN_CLICK_RESULT}\n{summary}\n{END_CLICK_RESULT}\n"
                output += click_result
                prompt += output
                total_tokens += len(click_result.split())
        
        else:
            finished = True
            break

    # Add max limit message if needed
    if not finished and (total_tokens >= MAX_TOKENS or total_interactions >= MAX_INTERACTIONS):
        output += f"\n{BEGIN_CLICK_RESULT}\nYou have reached the limit for clicking links.\n{END_CLICK_RESULT}\n\nOK, I will now provide the final information based on my collected information.\n\n**Final Information:**"
        prompt += output
        _, final_response = await generate_response(
            client=client,
            prompt=prompt,
            semaphore=semaphore,
            generate_mode="completion",
            temperature=args.temperature,
            top_p=args.top_p,
            max_tokens=512,
            repetition_penalty=1.2,
            top_k=args.top_k_sampling,
            min_p=args.min_p,
            model_name=args.model_name,
        )
        output += final_response

    return output, original_prompt


async def process_single_sequence(
    seq: Dict,
    client: AsyncOpenAI,
    aux_client: AsyncOpenAI,
    semaphore: asyncio.Semaphore,
    args: argparse.Namespace,
    search_cache: Dict,
    url_cache: Dict,
    batch_output_records: List[Dict],
) -> Dict:
    """Process a single sequence through its entire reasoning chain with MAX_TOKENS limit"""
    
    # Initialize token counter, initial value set to the token count of the prompt (simply using split() as an approximation)
    MAX_TOKENS = 20000
    total_tokens = len(seq['prompt'].split())
    
    # Initialize web explorer interactions list
    seq['web_explorer'] = []
    
    # First response uses chat completion
    formatted_prompt, response = await generate_response(
        client=client,
        model_name=args.model_name,
        prompt=seq['prompt'],
        semaphore=semaphore,
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
        repetition_penalty=args.repetition_penalty,
        top_k=args.top_k_sampling,
        min_p=args.min_p,
        stop=[END_SEARCH_QUERY],
    )
    
    # Update token count and sequence fields
    tokens_this_response = len(response.split())
    total_tokens += tokens_this_response
    
    seq['output'] += response.replace('</think>\n', '')
    seq['history'].append(response.replace('</think>\n', ''))
    seq['original_prompt'] = formatted_prompt
    seq['prompt'] = formatted_prompt + response.replace('</think>\n', '')
    
    while not seq['finished']:
        # Check if sequence is finished
        if not seq['output'].rstrip().endswith(END_SEARCH_QUERY):
            seq['finished'] = True
            break
        
        search_query = extract_between(response, BEGIN_SEARCH_QUERY, END_SEARCH_QUERY)

        if seq['search_count'] < args.max_search_limit and total_tokens < MAX_TOKENS:
            if search_query is None or len(search_query) <= 5: # Too short, invalid query
                continue

            if search_query in seq['executed_search_queries']:
                # If search query was already executed, append message and continue
                append_text = f"\n\n{BEGIN_SEARCH_RESULT}You have already searched for this query.{END_SEARCH_RESULT}\n\nOK, let me use the previously found information."
                seq['prompt'] += append_text
                seq['output'] += append_text
                seq['history'].append(append_text)
                seq['search_count'] += 1
                total_tokens += len(append_text.split())
                continue

            _, search_intent = await generate_response(
                client=aux_client,
                model_name=args.aux_model_name,
                prompt=get_search_intent_instruction(seq['output']),
                semaphore=semaphore,
            )

            # Execute search and subsequent operations (same as original logic)
            if search_query in search_cache:
                results = search_cache[search_query]
            else:
                try:
                    # results = bing_web_search(search_query, args.bing_subscription_key, args.bing_endpoint)
                    results = await bing_web_search_async(search_query, args.bing_subscription_key, args.bing_endpoint)
                    search_cache[search_query] = results
                except Exception as e:
                    print(f"Error during search query '{search_query}': {e}")
                    results = {}
            print(f'Searched for: "{search_query}"')

            relevant_info = extract_relevant_info(results)[:args.top_k]

            # Process documents
            urls_to_fetch = []
            for doc_info in relevant_info:
                url = doc_info['url']
                if url not in url_cache:
                    urls_to_fetch.append(url)

            if urls_to_fetch:
                try:
                    contents = await fetch_page_content_async(
                        urls_to_fetch, 
                        use_jina=args.use_jina, 
                        jina_api_key=args.jina_api_key, 
                        keep_links=args.keep_links
                    )
                    for url, content in contents.items():
                        # Only cache content if it doesn't contain error indicators
                        has_error = (any(indicator.lower() in content.lower() for indicator in error_indicators) and len(content.split()) < 64) or len(content) < 50 or len(content.split()) < 20
                        if not has_error:
                            url_cache[url] = content
                        # else:
                        #     print(f'---Fetching Error\n{content}')
                except Exception as e:
                    print(f"Error fetching URLs: {e}")

            # Get web page information for each result
            for doc_info in relevant_info:
                url = doc_info['url']
                if url not in url_cache:
                    raw_content = ""
                else:
                    raw_content = url_cache[url]
                    is_success, raw_content = extract_snippet_with_context(raw_content, doc_info['snippet'], context_chars=2000)

                # Check if content has error indicators
                has_error = any(indicator.lower() in raw_content.lower() for indicator in error_indicators) or raw_content == ""
            
                if has_error:
                    # If content has error, use it directly as summary
                    doc_info['page_info'] = "Can not fetch the page content."
                else:
                    # Use raw content directly as page info
                    doc_info['page_info'] = raw_content
                    # # Use detailed web page reader to process content
                    # reader_prompt = get_detailed_web_page_reader_instruction(search_query, search_intent, raw_content)
                    # _, page_info = await generate_response(
                    #     client=aux_client,
                    #     prompt=reader_prompt,
                    #     semaphore=semaphore,
                    #     max_tokens=4000,
                    #     model_name=args.aux_model_name,
                    # )
                    # doc_info['page_info'] = page_info

            formatted_documents = format_search_results(relevant_info)

            # Generate deep web exploration with interactions
            analysis, explorer_prompt = await generate_deep_web_explorer(
                client=client,
                aux_client=aux_client,
                search_query=search_query,
                search_intent=search_intent,
                document=formatted_documents,
                args=args,
                search_cache=search_cache,
                url_cache=url_cache,
                semaphore=semaphore,
            )

            extracted_info = extract_answer_fn(analysis, mode='summary')

            # Store web explorer input/output with all interactions
            seq['web_explorer'].append({
                "search_query": search_query,
                "Input": explorer_prompt,
                "Output": analysis,
                "Extracted_info": extracted_info
            })
            
            # Update sequence with search results
            append_text = f"\n\n{BEGIN_SEARCH_RESULT}{extracted_info}{END_SEARCH_RESULT}\n\n"
            seq['prompt'] += append_text
            seq['output'] += append_text
            seq['history'].append(append_text)
            
            seq['search_count'] += 1
            seq['executed_search_queries'].add(search_query)
            total_tokens += len(append_text.split())
            
            # Subsequent responses use completion mode
            _, response = await generate_response(
                client=client,
                model_name=args.model_name,
                prompt=seq['prompt'],
                semaphore=semaphore,
                temperature=args.temperature,
                top_p=args.top_p,
                max_tokens=args.max_tokens,
                repetition_penalty=args.repetition_penalty,
                top_k=args.top_k_sampling,
                min_p=args.min_p,
                stop=[END_SEARCH_QUERY],
                generate_mode="completion"
            )
            
            # Update token count and sequence fields
            tokens_this_response = len(response.split())
            total_tokens += tokens_this_response
            
            seq['output'] += response.replace('</think>\n', '')
            seq['history'].append(response.replace('</think>\n', ''))
            seq['prompt'] += response.replace('</think>\n', '')
            continue

        else:
            append_text = f"\n\n{BEGIN_SEARCH_RESULT}You have reached the search limit. You are not allowed to search.{END_SEARCH_RESULT}\n\n"
            seq['prompt'] += append_text
            seq['output'] += append_text
            seq['history'].append(append_text)
            
            _, final_response = await generate_response(
                client=client,
                prompt=seq['prompt'],
                semaphore=semaphore,
                temperature=args.temperature,
                top_p=args.top_p,
                max_tokens=args.max_tokens,
                repetition_penalty=1.2,
                top_k=args.top_k_sampling,
                min_p=args.min_p,
                model_name=args.model_name,
                generate_mode="completion"
            )
            
            seq['output'] += final_response
            seq['history'].append(final_response)
            seq['finished'] = True
            break
    
    return seq


async def load_lora_adapter(api_base_url: str, lora_name: str, lora_path: str) -> bool:
    """Load a LoRA adapter with the specified name and path"""
    try:
        lora_load_url = f"{api_base_url}/load_lora_adapter"
        lora_payload = {
            "lora_name": lora_name,
            "lora_path": lora_path
        }
        async with aiohttp.ClientSession() as session:
            async with session.post(lora_load_url, json=lora_payload) as response:
                return response.status == 200
    except Exception as e:
        print(f"Error loading LoRA adapter: {e}")
        return False

async def unload_lora_adapter(api_base_url: str, lora_name: str) -> bool:
    """Unload a LoRA adapter with the specified name"""
    try:
        unload_url = f"{api_base_url}/unload_lora_adapter"
        unload_payload = {"lora_name": lora_name}
        async with aiohttp.ClientSession() as session:
            async with session.post(unload_url, json=unload_payload) as response:
                return response.status == 200
    except Exception as e:
        print(f"Error unloading LoRA adapter: {e}")
        return False


async def main_async():
    args = parse_args()

    # Set random seed
    if args.seed is None:
        args.seed = int(time.time())
    random.seed(args.seed)
    np.random.seed(args.seed)

    if args.jina_api_key == 'None':
        jina_api_key = None

    # Modified data loading section
    if args.single_question:
        # Create a single item in the same format as dataset items
        filtered_data = [{
            'Question': args.single_question,
        }]
        args.dataset_name = 'custom'  # Set dataset name to custom for single questions
    else:
        # Original dataset loading logic
        if args.dataset_name == 'livecode':
            data_path = f'./data/LiveCodeBench/{args.split}.json'
        elif args.dataset_name == 'supergpqa':
            data_path = f'./data/SuperGPQA/{args.split}.json'
        elif args.dataset_name == 'webwalker':
            data_path = f'./data/WebWalkerQA/{args.split}.json'
        elif args.dataset_name == 'openthoughts':
            data_path = f'./data/OpenThoughts/{args.split}.json'
        elif args.dataset_name in ['math500', 'gpqa', 'aime', 'amc', 'gaia', 'hle', 'limo']:
            data_path = f'./data/{args.dataset_name.upper()}/{args.split}.json'
        else:
            data_path = f'./data/QA_Datasets/{args.dataset_name}.json'

        print('-----------------------')
        print(f'Using {args.dataset_name} {args.split} set.')
        print('-----------------------')

    # ---------------------- Caching Mechanism ----------------------
    cache_dir = './cache'
    search_cache_path = os.path.join(cache_dir, 'search_cache.json')
    if args.keep_links:
        url_cache_path = os.path.join(cache_dir, 'url_cache_with_links.json')
    else:
        url_cache_path = os.path.join(cache_dir, 'url_cache.json')

    os.makedirs(cache_dir, exist_ok=True)

    # Load existing caches
    search_cache = json.load(open(search_cache_path)) if os.path.exists(search_cache_path) else {}
    url_cache = json.load(open(url_cache_path)) if os.path.exists(url_cache_path) else {}

    def save_caches():
        with open(search_cache_path, 'w', encoding='utf-8') as f:
            json.dump(search_cache, f, ensure_ascii=False, indent=2)
        with open(url_cache_path, 'w', encoding='utf-8') as f:
            json.dump(url_cache, f, ensure_ascii=False, indent=2)

    # Define output directory
    if 'qwq' in args.model_name.lower():
        model_short_name = 'qwq'
    elif 'deepseek' in args.model_name.lower():
        if 'llama-8b' in args.model_name.lower():
            model_short_name = 'dpsk-llama-8b'
        elif 'llama-70b' in args.model_name.lower():
            model_short_name = 'dpsk-llama-70b'
        elif 'qwen-1.5b' in args.model_name.lower():
            model_short_name = 'dpsk-qwen-1.5b'
        elif 'qwen-7b' in args.model_name.lower():
            model_short_name = 'dpsk-qwen-7b'
        elif 'qwen-32b' in args.model_name.lower():
            model_short_name = 'dpsk-qwen-32b'
    elif 'sky-t1' in args.model_name.lower():
        model_short_name = 'sky-t1'
    else:
        model_short_name = args.model_name.split('/')[-1].lower().replace('-instruct', '')

    output_dir = f'./outputs/{args.dataset_name}.{model_short_name}.webthinker'
    os.makedirs(output_dir, exist_ok=True)

    # Initialize the OpenAI client
    client = AsyncOpenAI(
        api_key="empty",
        base_url=args.api_base_url,
    )
    # Initialize auxiliary client
    aux_client = AsyncOpenAI(
        api_key="empty",
        base_url=args.aux_api_base_url,
    )
    
    if not args.single_question:
        # Load and prepare data
        with open(data_path, 'r', encoding='utf-8') as json_file:
            filtered_data = json.load(json_file)

        if args.subset_num != -1:
            indices = list(range(len(filtered_data)))
            selected_indices = random.sample(indices, min(args.subset_num, len(indices)))
            filtered_data = [filtered_data[i] for i in selected_indices]

    # Prepare sequences
    active_sequences = []
    for item in filtered_data:
        question = item['Question']
        
        # Get appropriate instruction and user prompt based on dataset
        if args.dataset_name in ['nq', 'triviaqa', 'hotpotqa', 'musique', 'bamboogle', '2wiki', 'webwalker', 'gaia', 'hle', 'supergpqa']:
            if args.dataset_name in ['nq', 'triviaqa']:
                instruction = get_singleqa_search_o1_instruction(args.max_search_limit)
            else:
                instruction = get_multiqa_search_o1_instruction(args.max_search_limit)
            
            if 'qwq' in args.model_name.lower() or 'sky-t1' in args.model_name.lower():
                user_prompt = get_task_instruction_openqa(question, model_name='qwq')
            elif 'deepseek' in args.model_name.lower():
                user_prompt = get_task_instruction_openqa(question, model_name='dpsk')
            else:
                user_prompt = get_task_instruction_openqa(question)

        elif args.dataset_name in ['openthoughts']:
            if args.split == 'math':
                instruction = get_math_search_o1_instruction(args.max_search_limit)
                user_prompt = get_task_instruction_openqa(question, model_name='qwq')
            elif args.split == 'code':
                instruction = get_code_search_o1_instruction(args.max_search_limit)
                user_prompt = get_task_instruction_code(question, model_name='qwq')
            elif args.split == 'puzzle':
                instruction = get_singleqa_search_o1_instruction(args.max_search_limit)
                user_prompt = get_task_instruction_multi_choice(question, model_name='qwq')
            else:
                instruction = get_singleqa_search_o1_instruction(args.max_search_limit)
                user_prompt = get_task_instruction_openqa(question, model_name='qwq')

        elif args.dataset_name in []:
            instruction = get_gpqa_web_thinker_instruction(args.max_search_limit)
            # instruction = get_web_thinker_instruction()
            user_prompt = get_task_instruction_openqa(question, model_name='qwq')

        elif args.dataset_name in ['math500', 'aime', 'amc', 'limo']:
            instruction = get_math_search_o1_instruction(args.max_search_limit)
            if 'qwq' in args.model_name.lower() or 'sky-t1' in args.model_name.lower():
                user_prompt = get_task_instruction_math(question, model_name='qwq')
            elif 'deepseek' in args.model_name.lower():
                user_prompt = get_task_instruction_math(question, model_name='dpsk')
            else:
                user_prompt = get_task_instruction_math(question)

        elif args.dataset_name in ['gpqa']:
            instruction = get_gpqa_web_thinker_instruction(args.max_search_limit)
            if 'qwq' in args.model_name.lower() or 'sky-t1' in args.model_name.lower():
                user_prompt = get_task_instruction_multi_choice(question, model_name='qwq')
            elif 'deepseek' in args.model_name.lower():
                user_prompt = get_task_instruction_multi_choice(question, model_name='dpsk')
            elif 'llama' in args.model_name.lower():
                user_prompt = get_task_instruction_multi_choice(question, model_name='llama')
            else:
                user_prompt = get_task_instruction_multi_choice(question)

        elif args.dataset_name == 'livecode':
            instruction = get_code_search_o1_instruction(args.max_search_limit)
            question_title = item.get('question_title', '')
            if 'qwq' in args.model_name.lower() or 'deepseek' in args.model_name.lower() or 'sky-t1' in args.model_name.lower():
                user_prompt = get_task_instruction_code(question, question_title=question_title, model_name='qwq')
            else:
                user_prompt = get_task_instruction_code(question)
        else:
            instruction = get_multiqa_search_o1_instruction(args.max_search_limit)
            user_prompt = get_task_instruction_openqa(question)

        prompt = instruction + user_prompt
        item['prompt'] = prompt
        active_sequences.append({
            'item': item,
            'prompt': prompt,
            'output': '',
            'finished': False,
            'history': [],
            'search_count': 0,
            'executed_search_queries': set(),
        })

    # Initialize batch output records
    batch_output_records = []
    start_time = time.time()

    # Create semaphore for concurrent API calls
    semaphore = asyncio.Semaphore(args.concurrent_limit)

    # Load LoRA adapter if specified
    if args.lora_name and args.lora_path:
        print(f"Loading LoRA adapter '{args.lora_name}' from {args.lora_path}")
        success = await load_lora_adapter(args.api_base_url, args.lora_name, args.lora_path)
        if not success:
            print("Failed to load LoRA adapter")
            return
        else:
            print("LoRA adapter loaded successfully")

    try:
        # Process all sequences concurrently
        tasks = [
            process_single_sequence(
                seq=seq,
                client=client,
                aux_client=aux_client,
                semaphore=semaphore,
                args=args,
                search_cache=search_cache,
                url_cache=url_cache,
                batch_output_records=batch_output_records
            )
            for seq in active_sequences
        ]

        # Run all sequences concurrently with progress bar
        with tqdm(total=len(tasks)) as pbar:
            async def track_progress(task):
                result = await task
                pbar.update(1)
                return result
            
            tracked_tasks = [track_progress(task) for task in tasks]
            completed_sequences = await asyncio.gather(*tracked_tasks)
    finally:
        # Unload LoRA adapter if it was loaded
        if args.lora_name:
            print(f"Unloading LoRA adapter '{args.lora_name}'")
            await unload_lora_adapter(args.api_base_url, args.lora_name)
            print("LoRA adapter unloaded successfully")

    total_time = time.time() - start_time

    if args.eval:
        # Prepare output list and save results
        output_list = [seq['output'] for seq in completed_sequences]
        run_evaluation(filtered_data, [seq['original_prompt'] for seq in completed_sequences], output_list, args.dataset_name, output_dir, total_time, args.split)
    else:
        t = time.localtime()
        random_num = str(random.randint(0, 99)).zfill(2)
        result_json_name = f'{args.split}.{t.tm_mon}.{t.tm_mday},{t.tm_hour}:{t.tm_min}.{random_num}.json'
        if 'DPO' in args.model_name:
            result_json_name = f'{args.split}.{t.tm_mon}.{t.tm_mday},{t.tm_hour}:{t.tm_min}.{random_num}.dpo.json'
        elif 'SFT' in args.model_name:
            result_json_name = f'{args.split}.{t.tm_mon}.{t.tm_mday},{t.tm_hour}:{t.tm_min}.{random_num}.sft.json'
        
        for item, seq in zip(filtered_data, completed_sequences):
            item['prompt'] = seq['original_prompt']
            item['Output'] = seq['output']
            item['WebExplorer'] = seq['web_explorer']  # Updated field name
            
        with open(os.path.join(output_dir, result_json_name), mode='w', encoding='utf-8') as json_file:
            json.dump(filtered_data, json_file, indent=4, ensure_ascii=False)

    # Save caches
    save_caches()
    print("Process completed.")

def main():
    asyncio.run(main_async())

if __name__ == "__main__":
    main()
