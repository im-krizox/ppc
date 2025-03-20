import asyncio
import aiohttp

async def fetch(session, url, payload):
    async with session.post(url, json=payload) as response:
        return await response.json()

async def main():
    url = 'https://<OBFUSCATED_PROJECT_ID>.cloudfunctions.net/pokemon'
    async with aiohttp.ClientSession() as session:
        tasks = []
        for pokemon_id in range(1, 901):
            payload = {"id": pokemon_id}
            tasks.append(fetch(session, url, payload))
        
        responses = await asyncio.gather(*tasks)
        for response in responses:
            print(response)

if __name__ == '__main__':
    asyncio.run(main())
