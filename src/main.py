import discord
from configs.config import Settings
from discord.ext import commands
import asyncio

intents = discord.Intents.default()
intents.message_content = True 
bot = commands.Bot(intents=intents, command_prefix=Settings.DISCORDPREFIX)

@bot.event
async def on_ready():
    if not bot.synced:
        await bot.tree.sync()
        bot.synced = True
        print("✅ Synced commands.")
    print(f"🤖 Logged in as {bot.user.name}")

async def main():
    
    await bot.load_extension('cogs.commands')
    await bot.load_extension('cogs.commands_context_menu')
    await bot.load_extension('cogs.commands_slash')  
    await bot.start(Settings.DISCORDTOKEN)

if __name__ == "__main__":
    asyncio.run(main())
