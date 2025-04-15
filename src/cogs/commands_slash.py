# cogs/slash_commands.py
import discord
from discord.ext import commands
from discord import app_commands

class SlashCommands(commands.Cog):
    def __init__(self, bot: commands.Bot):
        self.bot = bot

    @app_commands.command(name="say", description="打招呼")
    @app_commands.describe(name="你要叫誰", times="說幾次")
    async def say(self, interaction: discord.Interaction, name: str, times: int):
        await interaction.response.send_message(f"Hello {name}!\n" * times)

    @app_commands.command(name="ping", description="Check bot latency")
    async def ping(self, interaction: discord.Interaction):
        latency = round(self.bot.latency * 1000)
        await interaction.response.send_message(f"Pong! `{latency} ms`")

async def setup(bot):
    await bot.add_cog(SlashCommands(bot))
