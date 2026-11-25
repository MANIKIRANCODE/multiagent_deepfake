
import cv2
import time

from core.video_stream import VideoStream
from core.audio_stream import AudioStream
from core.orchestrator import MultiAgentOrchestrator


def draw_status_panel(frame, agents, decision, fps):
    height, width = frame.shape[:2]

    # Create a translucent status panel
    panel_width = min(430, width - 20)
    panel_height = 285

    overlay = frame.copy()

    cv2.rectangle(
        overlay,
        (10, 10),
        (10 + panel_width, 10 + panel_height),
        (25, 25, 25),
        -1
    )

    frame[:] = cv2.addWeighted(overlay, 0.78, frame, 0.22, 0)

    # Heading
    cv2.putText(
        frame,
        "MULTI-AGENT ANTI-SPOOFING",
        (22, 38),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.58,
        (0, 255, 255),
        2
    )

    cv2.putText(
        frame,
        f"FPS: {fps:.1f}",
        (22, 65),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48,
        (255, 255, 255),
        1
    )

    # Agent statuses
    y = 100

    for agent in agents:
        name = agent.get("agent", "Unknown Agent")
        status = str(agent.get("status", "UNKNOWN"))

        score = agent.get("score")

        if "NOT_CONFIGURED" in status:
            color = (0, 165, 255)
        elif "ERROR" in status or "FAILED" in status:
            color = (0, 0, 255)
        elif "OK" in status or "DETECTED" in status:
            color = (0, 255, 0)
        else:
            color = (255, 255, 255)

        message = f"{name}: {status}"

        if score is not None:
            message += f" | {score}"

        cv2.putText(
            frame,
            message[:55],
            (22, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.43,
            color,
            1
        )

        y += 27

    # Final assessment
    y += 10

    risk = decision.get("status", "INSUFFICIENT_EVIDENCE")

    cv2.putText(
        frame,
        "FINAL ASSESSMENT:",
        (22, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48,
        (255, 255, 255),
        1
    )

    cv2.putText(
        frame,
        str(risk)[:40],
        (22, y + 27),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (0, 255, 255),
        2
    )

    cv2.putText(
        frame,
        "Research prototype - not an identity verdict",
        (22, y + 52),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.35,
        (200, 200, 200),
        1
    )


def main():

    video = VideoStream()
    audio = AudioStream()
    orchestrator = MultiAgentOrchestrator()

    try:
        video.start()
    except Exception as exc:
        print(f"Webcam error: {exc}")
        return

    audio.start()

    print("Multi-Agent Anti-Spoofing System Started")
    print("Press Q to quit")

    previous = time.time()
    fps = 0.0

    try:
        while True:

            ok, frame = video.read()

            if not ok:
                print("Could not read webcam frame.")
                break

            now = time.time()
            dt = now - previous
            previous = now

            if dt > 0:
                current_fps = 1.0 / dt
                fps = (
                    0.9 * fps + 0.1 * current_fps
                    if fps else current_fps
                )

            output = orchestrator.analyze_frame(
                frame,
                audio_level=audio.level,
                audio_available=audio.available,
                audio_samples=audio.get_samples()
            )

            agents = output.get("agents", [])

            # Draw detected face boxes
            for agent in agents:
                for x, y, w, h in agent.get("faces", []):
                    cv2.rectangle(
                        frame,
                        (x, y),
                        (x + w, y + h),
                        (0, 255, 0),
                        2
                    )

            decision = output.get(
                "decision",
                {"status": "INSUFFICIENT_EVIDENCE"}
            )

            draw_status_panel(
                frame,
                agents,
                decision,
                fps
            )

            cv2.imshow(
                "Multi-Agent Anti-Spoofing System",
                frame
            )

            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                break

    except Exception as exc:
        print(f"Analysis error: {exc}")

    finally:
        audio.stop()
        video.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
