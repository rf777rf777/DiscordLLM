# cogs/context_menu.py
import discord
from discord.ext import commands

class ContextMenus(commands.Cog):
    def __init__(self, bot: commands.Bot):
        self.bot = bot

    @discord.app_commands.context_menu(name="Detect Images")
    async def detect_image_context(self, interaction: discord.Interaction, message: discord.Message):
        await interaction.response.send_message(f"You selected a message with: {len(message.attachments)} attachment(s)", ephemeral=True)

async def setup(bot):
    await bot.add_cog(ContextMenus(bot))
