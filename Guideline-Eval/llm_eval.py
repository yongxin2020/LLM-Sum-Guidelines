import openai
from openai import OpenAI
import json
import argparse
import tqdm
import time

if __name__ == '__main__':

    argparser = argparse.ArgumentParser()
    argparser.add_argument('--prompt_fp', type=str, required=True)
    argparser.add_argument('--save_fp', type=str, required=True)
    argparser.add_argument('--input_fp', type=str, required=True, help="Input JSON file with evaluation samples")
    argparser.add_argument('--key', type=str, required=True)
    argparser.add_argument('--model', type=str, required=True, help="Model name, e.g., gpt-4-0613, deepseek-reasoner")
    argparser.add_argument('--base_url', type=str, default=None, help="API base URL (set for non-OpenAI endpoints)")
    args = argparser.parse_args()

    # Set up client
    if args.base_url:
        client = OpenAI(api_key=args.key, base_url=args.base_url)
    else:
        client = OpenAI(api_key=args.key)

    eval_data = json.load(open(args.input_fp))
    prompt = open(args.prompt_fp).read()

    max_retries = 3
    retry_delay = 2

    new_json = []
    ignore = 0

    for instance in tqdm.tqdm(eval_data):
        source = instance['dialogue']
        system_output = instance['system_output']
        cur_prompt = prompt.replace('{{Document}}', source).replace('{{Summary}}', system_output)
        instance['prompt'] = cur_prompt

        for attempt in range(max_retries):
            try:
                _response = client.chat.completions.create(
                    model=args.model,
                    messages=[
                        {"role": "system", "content": "You are a helpful assistant"},
                        {"role": "user", "content": cur_prompt},
                              ],
                    #temperature=2,
                    #max_tokens=5,
                    #top_p=1,
                    #frequency_penalty=0,
                    #presence_penalty=0,
                    #stop=None,
                    # logprobs=40,
                    stream=False
                )
                time.sleep(0.5)
                response = _response.choices[0].message.content
                instance['response'] = response
                new_json.append(instance)
                break
            except Exception as e:
                print(f"Retry {attempt + 1} failed for instance {instance.get('id', '')}: {e}")
                time.sleep(retry_delay)
        else:
            print(f"Max retries reached for instance: {instance.get('id', '')}")
            ignore += 1
            continue

    print('ignored total', ignore)

    with open(args.save_fp, 'w', encoding="utf-8") as f:
        json.dump(new_json, f, ensure_ascii=False, indent=4)
