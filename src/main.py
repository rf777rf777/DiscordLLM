import discord
from configs.config import Settings
from discord.ext import commands

if __name__ == "__main__":
    print(Settings().OPENAIAPIKEY)
    intents = discord.Intents.default()

    intents.message_content = True 
    bot = commands.Bot(intents=intents, command_prefix=Settings.DISCORDPREFIX)
  
    #on_ready event
    @bot.event
    async def on_ready():
        print(f'Logged in as {bot.user.name}')
        try:
            await bot.load_extension('cogs.commands')
            await bot.load_extension('cogs.events')
            synced = await bot.tree.sync()
            print(f"Synced {len(synced)} command(s) across all guilds.")
        except Exception as e:
            print(e)
  
    bot.run(Settings.DISCORDTOKEN)