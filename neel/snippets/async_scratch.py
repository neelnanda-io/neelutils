# # %%
# import asyncio
# from transformers import AutoModelForCausalLM

# async def gpt2m():
#     print("Start M")
#     start_time = time.time()
#     got = await AutoModelForCausalLM.from_pretrained("gpt2-medium")
#     print("Done M", time.time() - start_time)
#     return got


# async def gpt2l():
#     print("Start L")
#     start_time = time.time()
#     got = await AutoModelForCausalLM.from_pretrained("gpt2-large")
#     print("Done L", time.time() - start_time)
#     return got


# async def main():
#     print("Starting!")
#     out = await asyncio.gather(gpt2m(), gpt2l())
#     print(out)
# import time
# start_time = time.time()
# print("Start Time")
# asyncio.run(main())
# print(time.time() - start_time)

# # %%
