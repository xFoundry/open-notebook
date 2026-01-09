import subprocess
import time
import uuid
from pathlib import Path
from typing import Optional, Union

from loguru import logger
from pydantic import BaseModel
from surreal_commands import CommandInput, CommandOutput, command

from open_notebook.config import DATA_FOLDER
from open_notebook.database.repository import ensure_record_id, repo_query
from open_notebook.podcasts.models import EpisodeProfile, PodcastEpisode, SpeakerProfile

try:
    from podcast_creator import configure, create_podcast
    import podcast_creator.core as podcast_core
except ImportError as e:
    logger.error(f"Failed to import podcast_creator: {e}")
    raise ValueError("podcast_creator library not available")


# =============================================================================
# MONKEY-PATCH: Replace moviepy-based audio concatenation with direct FFMPEG
# This fixes the "[Errno 11] Resource temporarily unavailable" error that occurs
# when moviepy tries to load 60+ audio clips simultaneously, exhausting file
# descriptors. Direct FFMPEG concatenation uses minimal resources.
# =============================================================================
async def combine_audio_files_ffmpeg(
    audio_dir: Union[Path, str], final_filename: str, final_output_dir: Union[Path, str]
):
    """
    Combines multiple audio files into a single MP3 file using direct FFMPEG.
    This replaces the moviepy-based version which exhausts file descriptors.
    """
    logger.info("[Patched] combine_audio_files_ffmpeg called - using direct FFMPEG")

    if isinstance(audio_dir, str):
        audio_dir = Path(audio_dir)
    if isinstance(final_output_dir, str):
        final_output_dir = Path(final_output_dir)

    list_of_audio_paths = sorted(audio_dir.glob("*.mp3"))

    if not list_of_audio_paths:
        logger.warning("combine_audio_files: No audio files found in directory.")
        return {"combined_audio_path": "ERROR: No audio segment data"}

    logger.info(f"Found {len(list_of_audio_paths)} audio clips to combine")

    # Create output directory
    final_output_dir.mkdir(parents=True, exist_ok=True)

    # Determine output filename
    if final_filename and isinstance(final_filename, str):
        output_filename = Path(final_filename).name
        if not output_filename.endswith(".mp3"):
            output_filename += ".mp3"
    else:
        output_filename = f"combined_{uuid.uuid4().hex}.mp3"
        logger.warning(f"'final_filename' not provided. Using: {output_filename}")

    output_path = final_output_dir / output_filename

    # Create a temporary file list for FFMPEG concat demuxer
    filelist_path = audio_dir / "filelist.txt"
    try:
        with open(filelist_path, "w") as f:
            for audio_path in list_of_audio_paths:
                # FFMPEG concat requires single quotes for paths with special chars
                escaped_path = str(audio_path.resolve()).replace("'", "'\\''")
                f.write(f"file '{escaped_path}'\n")

        logger.info(f"Created FFMPEG file list with {len(list_of_audio_paths)} entries")

        # Use FFMPEG concat demuxer - much more efficient than moviepy
        # -f concat: use concat demuxer
        # -safe 0: allow absolute paths
        # -i filelist.txt: input file list
        # -c copy: stream copy (no re-encoding, very fast)
        # For MP3 we need to re-encode to ensure proper concatenation
        cmd = [
            "ffmpeg",
            "-y",  # Overwrite output file
            "-f", "concat",
            "-safe", "0",
            "-i", str(filelist_path),
            "-c:a", "libmp3lame",  # Re-encode as MP3
            "-q:a", "2",  # High quality
            str(output_path)
        ]

        logger.info(f"Running FFMPEG: {' '.join(cmd)}")

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=600  # 10 minute timeout for large files
        )

        if result.returncode != 0:
            logger.error(f"FFMPEG failed with code {result.returncode}")
            logger.error(f"FFMPEG stderr: {result.stderr}")
            return {"combined_audio_path": f"ERROR: FFMPEG failed - {result.stderr[:500]}"}

        # Get duration using ffprobe
        duration = 0.0
        try:
            probe_cmd = [
                "ffprobe", "-v", "error",
                "-show_entries", "format=duration",
                "-of", "default=noprint_wrappers=1:nokey=1",
                str(output_path)
            ]
            probe_result = subprocess.run(probe_cmd, capture_output=True, text=True)
            if probe_result.returncode == 0:
                duration = float(probe_result.stdout.strip())
        except Exception as e:
            logger.warning(f"Could not get duration: {e}")

        logger.info(f"Successfully combined audio to: {output_path.resolve()}")
        return {
            "combined_audio_path": str(output_path.resolve()),
            "original_segments_count": len(list_of_audio_paths),
            "total_duration_seconds": duration,
        }

    except subprocess.TimeoutExpired:
        logger.error("FFMPEG timed out after 10 minutes")
        return {"combined_audio_path": "ERROR: FFMPEG timeout"}
    except Exception as e:
        logger.error(f"Error combining audio files: {e}")
        return {"combined_audio_path": f"ERROR: {e}"}
    finally:
        # Clean up filelist
        if filelist_path.exists():
            try:
                filelist_path.unlink()
            except Exception:
                pass


# Apply the monkey-patch
podcast_core.combine_audio_files = combine_audio_files_ffmpeg
logger.info("Applied FFMPEG-based audio concatenation patch to podcast_creator")


def full_model_dump(model):
    if isinstance(model, BaseModel):
        return model.model_dump()
    elif isinstance(model, dict):
        return {k: full_model_dump(v) for k, v in model.items()}
    elif isinstance(model, list):
        return [full_model_dump(item) for item in model]
    else:
        return model


class PodcastGenerationInput(CommandInput):
    episode_profile: str
    speaker_profile: str
    episode_name: str
    content: str
    briefing_suffix: Optional[str] = None


class PodcastGenerationOutput(CommandOutput):
    success: bool
    episode_id: Optional[str] = None
    audio_file_path: Optional[str] = None
    transcript: Optional[dict] = None
    outline: Optional[dict] = None
    processing_time: float
    error_message: Optional[str] = None


@command("generate_podcast", app="open_notebook")
async def generate_podcast_command(
    input_data: PodcastGenerationInput,
) -> PodcastGenerationOutput:
    """
    Real podcast generation using podcast-creator library with Episode Profiles
    """
    start_time = time.time()

    try:
        logger.info(
            f"Starting podcast generation for episode: {input_data.episode_name}"
        )
        logger.info(f"Using episode profile: {input_data.episode_profile}")

        # 1. Load Episode and Speaker profiles from SurrealDB
        episode_profile = await EpisodeProfile.get_by_name(input_data.episode_profile)
        if not episode_profile:
            raise ValueError(
                f"Episode profile '{input_data.episode_profile}' not found"
            )

        speaker_profile = await SpeakerProfile.get_by_name(
            episode_profile.speaker_config
        )
        if not speaker_profile:
            raise ValueError(
                f"Speaker profile '{episode_profile.speaker_config}' not found"
            )

        logger.info(f"Loaded episode profile: {episode_profile.name}")
        logger.info(f"Loaded speaker profile: {speaker_profile.name}")

        # 3. Load all profiles and configure podcast-creator
        episode_profiles = await repo_query("SELECT * FROM episode_profile")
        speaker_profiles = await repo_query("SELECT * FROM speaker_profile")

        # Transform the surrealdb array into a dictionary for podcast-creator
        episode_profiles_dict = {
            profile["name"]: profile for profile in episode_profiles
        }
        speaker_profiles_dict = {
            profile["name"]: profile for profile in speaker_profiles
        }

        # 4. Generate briefing
        briefing = episode_profile.default_briefing
        if input_data.briefing_suffix:
            briefing += f"\n\nAdditional instructions: {input_data.briefing_suffix}"

        # Create the a record for the episose and associate with the ongoing command
        episode = PodcastEpisode(
            name=input_data.episode_name,
            episode_profile=full_model_dump(episode_profile.model_dump()),
            speaker_profile=full_model_dump(speaker_profile.model_dump()),
            command=ensure_record_id(input_data.execution_context.command_id)
            if input_data.execution_context
            else None,
            briefing=briefing,
            content=input_data.content,
            audio_file=None,
            transcript=None,
            outline=None,
        )
        await episode.save()

        configure("speakers_config", {"profiles": speaker_profiles_dict})
        configure("episode_config", {"profiles": episode_profiles_dict})

        logger.info("Configured podcast-creator with episode and speaker profiles")

        logger.info(f"Generated briefing (length: {len(briefing)} chars)")

        # 5. Create output directory
        output_dir = Path(f"{DATA_FOLDER}/podcasts/episodes/{input_data.episode_name}")
        output_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Created output directory: {output_dir}")

        # 6. Generate podcast using podcast-creator
        logger.info("Starting podcast generation with podcast-creator...")

        result = await create_podcast(
            content=input_data.content,
            briefing=briefing,
            episode_name=input_data.episode_name,
            output_dir=str(output_dir),
            speaker_config=speaker_profile.name,
            episode_profile=episode_profile.name,
        )

        episode.audio_file = (
            str(result.get("final_output_file_path")) if result else None
        )
        episode.transcript = {
            "transcript": full_model_dump(result["transcript"]) if result else None
        }
        episode.outline = full_model_dump(result["outline"]) if result else None
        await episode.save()

        processing_time = time.time() - start_time
        logger.info(
            f"Successfully generated podcast episode: {episode.id} in {processing_time:.2f}s"
        )

        return PodcastGenerationOutput(
            success=True,
            episode_id=str(episode.id),
            audio_file_path=str(result.get("final_output_file_path"))
            if result
            else None,
            transcript={"transcript": full_model_dump(result["transcript"])}
            if result.get("transcript")
            else None,
            outline=full_model_dump(result["outline"])
            if result.get("outline")
            else None,
            processing_time=processing_time,
        )

    except Exception as e:
        processing_time = time.time() - start_time
        logger.error(f"Podcast generation failed: {e}")
        logger.exception(e)

        # Check for specific GPT-5 extended thinking issue
        error_msg = str(e)
        if "Invalid json output" in error_msg or "Expecting value" in error_msg:
            # This often happens with GPT-5 models that use extended thinking (<think> tags)
            # and put all output inside thinking blocks
            error_msg += (
                "\n\nNOTE: This error commonly occurs with GPT-5 models that use extended thinking. "
                "The model may be putting all output inside <think> tags, leaving nothing to parse. "
                "Try using gpt-4o, gpt-4o-mini, or gpt-4-turbo instead in your episode profile."
            )

        return PodcastGenerationOutput(
            success=False, processing_time=processing_time, error_message=error_msg
        )
