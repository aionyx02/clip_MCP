// Plays a cut in the browser as it is edited, from the server's description of it: the small
// copies of its files, sequenced on a canvas, and their sound mixed with Web Audio. Every number
// comes from the server, worked out the way a render works it out; this only follows it. Where
// the browser can only come close (colour, music ducking by loudness) the description says so.
"use strict";

(() => {
  // How far ahead a clip's file is opened and put at its first frame, so it is ready on its cut.
  // These five are provisional: not yet tried on a slow computer.
  const PRELOAD_SECONDS = 2;
  // How far a file may run from the clock before it is put back by seeking rather than nudged.
  const SEEK_DRIFT = 0.5;
  // How far it may run before it is nudged faster or slower to catch up.
  const NUDGE_DRIFT = 0.04;
  // The canvas's shorter side: enough for a 480p copy, without drawing 4K every frame.
  const CANVAS_SHORT_SIDE = 720;
  // How many files not playing anything are kept open, to be used again without opening them.
  const KEEP_OPEN = 8;
  // How much darker the blurred copy behind a shot shown whole is: the render's `WHOLE_DIM`.
  const WHOLE_DIM = -0.12;
  const FONTS = '"Microsoft JhengHei", "PingFang TC", "Noto Sans TC", sans-serif';

  const dbToGain = (db) => Math.pow(10, (db || 0) / 20);
  const clamp = (value, low, high) => Math.max(low, Math.min(high, value));
  // How much picture an incoming clip shows before its in point: a dip shows it for half.
  const runUp = (clip) => !clip.transition ? 0 : clip.transition.kind === "dip" ? clip.transition.seconds / 2 : clip.transition.seconds;
  const pictureStart = (clip) => clip.timeline_in - runUp(clip);
  const soundStart = (clip) => clip.timeline_in - clip.audio_lead;
  const soundEnd = (clip) => clip.timeline_out + clip.audio_lag;
  const firstNeeded = (clip) => Math.min(pictureStart(clip), soundStart(clip));
  const lastNeeded = (clip) => Math.max(clip.timeline_out, soundEnd(clip));
  // Where in its file a clip is at a time on the timeline; negative before the file starts.
  // How close in a photo is and where across it the view sits, part of the way through its clip:
  // the render's `stills.motion_at`.
  const motionAt = (clip, progress) => {
    const moved = clamp(progress, 0, 1), amount = clip.motion_amount || 0;
    switch (clip.motion) {
      case "push": return [1 + amount * moved, 0.5];
      case "pull": return [1 + amount * (1 - moved), 0.5];
      case "pan_left": return [1 + amount, moved];
      case "pan_right": return [1 + amount, 1 - moved];
      default: return [1, 0.5];
    }
  };
  const sourceAt = (clip, time) => clip.source_start + (time - clip.timeline_in) * clip.speed;
  // Where its repaired sound starts in the file: the render cuts it from the same place.
  const repairedFrom = (clip) => Math.max(0, clip.source_start - clip.audio_lead * clip.speed);

  class Player {
    constructor(stage, onState) {
      this.onState = onState || (() => {});
      this.canvas = document.createElement("canvas");
      this.canvas.className = "live";
      this.ctx = this.canvas.getContext("2d");
      this.shelf = document.createElement("div");
      this.shelf.className = "live-media";
      stage.prepend(this.canvas, this.shelf);
      this.described = null;
      this.time = 0;
      this.paused = true;
      this.slots = new Map(); // a clip being played: its elements, by track and clip ID
      this.free = []; // elements not playing anything, kept to be used again
      this.audio = null; // made on the first play: browsers only allow sound after a click
      this.dirty = true;
      this.frame = requestAnimationFrame(() => this.tick());
    }

    get ready() { return Boolean(this.described && this.described.duration > 0); }
    get duration() { return this.described?.duration || 0; }

    load(described) {
      this.described = described;
      const k = Math.min(1, CANVAS_SHORT_SIDE / Math.min(described.width, described.height));
      this.scale = k;
      this.canvas.width = Math.round(described.width * k);
      this.canvas.height = Math.round(described.height * k);
      this.time = clamp(this.time, 0, described.duration);
      this.held = null;
      this.dirty = true;
    }

    play() {
      if (!this.ready) return;
      this.mixer();
      if (this.audio.state === "suspended") this.audio.resume();
      if (this.time >= this.duration - 0.01) this.time = 0;
      this.paused = false;
      this.from = { time: this.time, at: performance.now() };
      this.onState();
    }

    pause() {
      if (this.paused) return;
      this.paused = true;
      for (const slot of this.slots.values()) for (const element of slot.elements) element.pause();
      this.dirty = true;
      this.onState();
    }

    seek(time) {
      this.time = clamp(time, 0, this.duration);
      if (!this.paused) this.from = { time: this.time, at: performance.now() };
      this.dirty = true;
    }

    show(visible) { this.canvas.hidden = !visible; if (!visible) this.pause(); }

    mute(silent) {
      this.silent = silent;
      if (this.out) this.out.gain.setTargetAtTime(silent ? 0 : 1, this.audio.currentTime, 0.02);
    }

    destroy() {
      cancelAnimationFrame(this.frame);
      for (const element of [...this.free, ...[...this.slots.values()].flatMap((slot) => slot.elements)]) {
        element.pause();
        element.removeAttribute("src");
        element.load();
      }
      this.audio?.close();
      this.canvas.remove();
      this.shelf.remove();
    }

    // ---------------------------------------------------------------- sound

    mixer() {
      if (this.audio) return;
      this.audio = new AudioContext({ latencyHint: "playback" });
      // A render brings the whole mix to one loudness and stops it from clipping; here the
      // measured gain stands in for the first and a limiter for the second.
      this.master = this.audio.createGain();
      const limiter = this.audio.createDynamicsCompressor();
      limiter.threshold.value = -1.5;
      limiter.knee.value = 0;
      limiter.ratio.value = 20;
      limiter.attack.value = 0.002;
      limiter.release.value = 0.08;
      // Last of all, on or off: two versions compared side by side are heard one at a time.
      this.out = this.audio.createGain();
      this.out.gain.value = this.silent ? 0 : 1;
      this.master.connect(limiter).connect(this.out).connect(this.audio.destination);
      this.buses = new Map();
    }

    bus(trackId) {
      if (!this.buses.has(trackId)) {
        const gain = this.audio.createGain();
        gain.connect(this.master);
        this.buses.set(trackId, gain);
      }
      return this.buses.get(trackId);
    }

    // An element's sound goes through the mixer once there is one; until then it is silent.
    route(element, trackId) {
      if (!this.audio) { element.muted = true; return; }
      if (!element.node) {
        element.node = this.audio.createMediaElementSource(element);
        element.level = this.audio.createGain();
        element.node.connect(element.level);
        element.muted = false;
      }
      if (element.track !== trackId) {
        element.level.disconnect();
        element.level.connect(this.bus(trackId));
        element.track = trackId;
      }
    }

    setLevel(element, value) {
      if (element.level) element.level.gain.setTargetAtTime(value, this.audio.currentTime, 0.015);
    }

    // The music's level at a time, lowered wherever somebody talks, exactly as a render ramps it.
    duckAt(time) {
      const duck = this.described.duck;
      if (!duck.spans) return 1;
      if (!this.held) {
        this.held = [];
        for (const [start, end] of [...duck.spans].sort((a, b) => a[0] - b[0])) {
          const last = this.held[this.held.length - 1];
          if (last && start - last[1] < duck.hold) last[1] = Math.max(last[1], end);
          else this.held.push([start, end]);
        }
      }
      let ramps = 0;
      for (const [start, end] of this.held) {
        ramps += clamp((time - (start - duck.attack)) / duck.attack, 0, 1) * clamp((end + duck.release - time) / duck.release, 0, 1);
      }
      return 1 - (1 - dbToGain(-duck.depth_db)) * ramps;
    }

    // How loud a clip's sound is at a time: its level, its fades, and nothing outside it.
    soundLevel(clip, time) {
      const start = soundStart(clip), end = soundEnd(clip);
      if (!clip.has_audio || time < start || time >= end) return 0;
      let level = clip.volume * dbToGain(clip.gain_db);
      if (clip.audio_fade_in > 0) level *= clamp((time - start) / clip.audio_fade_in, 0, 1);
      if (clip.audio_fade_out > 0) level *= clamp((end - time) / clip.audio_fade_out, 0, 1);
      return level;
    }

    // ---------------------------------------------------------------- the clips being played

    element(kind, src) {
      const index = this.free.findIndex((element) => element.tagName === kind && element.getAttribute("src") === src);
      if (index >= 0) return this.free.splice(index, 1)[0];
      const reuse = this.free.findIndex((element) => element.tagName === kind);
      let element = reuse >= 0 ? this.free.splice(reuse, 1)[0] : null;
      if (!element) {
        element = document.createElement(kind.toLowerCase());
        element.preload = "auto";
        element.playsInline = true;
        element.muted = true;
        element.addEventListener("seeked", () => { this.dirty = true; });
        element.addEventListener("loadeddata", () => { this.dirty = true; });
        this.shelf.append(element);
      }
      element.src = src;
      return element;
    }

    // A photo, loaded once and kept: it is one picture, with no time in it to follow.
    photo(src) {
      this.photos = this.photos || new Map();
      let image = this.photos.get(src);
      if (!image) {
        image = new Image();
        image.decoding = "async";
        image.addEventListener("load", () => { this.dirty = true; });
        image.src = src;
        this.photos.set(src, image);
      }
      return image;
    }

    // Open the files of the clips playing now or soon, and let go of the rest.
    gather() {
      const wanted = new Map();
      const from = this.time - 0.1, to = this.time + (this.paused ? 0.1 : PRELOAD_SECONDS);
      for (const track of this.described.tracks) {
        for (const clip of track.clips) {
          if (!clip.proxy || lastNeeded(clip) < from || firstNeeded(clip) > to) continue;
          wanted.set(`${track.id}/${clip.id}`, { track, clip });
        }
      }
      for (const [key, slot] of this.slots) {
        const now = wanted.get(key);
        if (now && now.clip.proxy === slot.clip.proxy && now.clip.repaired === slot.clip.repaired) {
          slot.clip = now.clip; slot.track = now.track;
          continue;
        }
        for (const element of slot.elements) { element.pause(); this.setLevel(element, 0); this.free.push(element); }
        this.slots.delete(key);
      }
      // A few are kept open to be used again; the rest are closed.
      while (this.free.length > KEEP_OPEN) {
        const element = this.free.shift();
        element.removeAttribute("src");
        element.load();
        element.remove();
      }
      for (const [key, { track, clip }] of wanted) {
        if (this.slots.has(key)) continue;
        if (clip.still) {
          this.slots.set(key, { track, clip, picture: this.photo(clip.proxy), sound: null, elements: [] });
          continue;
        }
        const picture = this.element("VIDEO", clip.proxy);
        const sound = clip.repaired ? this.element("AUDIO", clip.repaired) : null;
        this.slots.set(key, { track, clip, picture, sound, elements: sound ? [picture, sound] : [picture] });
      }
    }

    // Keep one element where the clock says it should be: seek when far off, nudge when close.
    follow(element, target, speed, live, pitch) {
      if ("preservesPitch" in element) element.preservesPitch = pitch;
      if (element.readyState < 1) return;
      target = clamp(target, 0, element.duration || target);
      const off = target - element.currentTime;
      if (!live) {
        if (!element.paused) element.pause();
        if (!element.seeking && Math.abs(off) > 0.02) element.currentTime = target;
        return;
      }
      if (Math.abs(off) > SEEK_DRIFT && !element.seeking) element.currentTime = target;
      element.playbackRate = Math.abs(off) > NUDGE_DRIFT ? speed * (1 + clamp(off * 0.5, -0.1, 0.1)) : speed;
      if (element.paused) element.play().catch(() => {});
    }

    // Whether everything that has to be seen or heard now has arrived; the clock waits until it has.
    arrived() {
      for (const slot of this.slots.values()) {
        const { clip } = slot;
        if (this.time < firstNeeded(clip) || this.time >= lastNeeded(clip)) continue;
        if (clip.still && !slot.picture.complete) return false;
        for (const element of slot.elements) if (!element.ended && (element.readyState < 3 || element.seeking)) return false;
      }
      return true;
    }

    tick() {
      this.frame = requestAnimationFrame(() => this.tick());
      if (!this.ready || this.canvas.hidden) return;
      if (!this.paused) {
        if (this.arrived()) {
          this.time = this.from.time + (performance.now() - this.from.at) / 1000;
        } else {
          this.from = { time: this.time, at: performance.now() };
        }
        if (this.time >= this.duration) {
          this.time = this.duration;
          this.pause();
        }
      }
      this.gather();
      const time = this.time;
      if (this.audio) this.master.gain.setTargetAtTime(dbToGain(this.described.gain_db), this.audio.currentTime, 0.05);
      for (const slot of this.slots.values()) {
        const { clip, track, picture, sound } = slot;
        if (clip.still) continue;
        if (clip.card) {
          this.route(picture, track.id);
          if (this.audio) this.setLevel(picture, 0);
          this.follow(picture, clip.card.at, 1, false, true);
          continue;
        }
        const live = !this.paused && time >= firstNeeded(clip) - 0.05 && time < lastNeeded(clip);
        const heard = sound || picture;
        this.route(picture, track.id);
        if (sound) this.route(sound, track.id);
        if (this.audio) {
          this.setLevel(heard, live ? this.soundLevel(clip, time) : 0);
          if (sound) this.setLevel(picture, 0);
          if (track.duck) this.bus(track.id).gain.setTargetAtTime(this.duckAt(time), this.audio.currentTime, 0.02);
        }
        const at = Math.max(firstNeeded(clip), time);
        this.follow(picture, sourceAt(clip, at), clip.speed, live, clip.preserve_pitch);
        if (sound) this.follow(sound, sourceAt(clip, at) - repairedFrom(clip), clip.speed, live, clip.preserve_pitch);
      }
      if (!this.paused || this.dirty) {
        this.dirty = false;
        this.draw(time);
      }
    }

    // ---------------------------------------------------------------- picture

    draw(time) {
      const { ctx } = this;
      const k = this.scale;
      ctx.setTransform(1, 0, 0, 1, 0, 0);
      ctx.globalAlpha = 1;
      ctx.filter = "none";
      ctx.fillStyle = "#000";
      ctx.fillRect(0, 0, this.canvas.width, this.canvas.height);
      ctx.setTransform(k, 0, 0, k, 0, 0);
      for (const track of this.described.tracks) {
        if (track.type !== "video") continue;
        const showing = track.clips.find((clip) => time >= clip.timeline_in && time < clip.timeline_out);
        if (showing) this.drawClip(track, showing, time);
        // A transition runs the next clip in over the end of this one, and ends on the cut.
        const incoming = track.clips.find((clip) => clip.transition && showing
          && Math.abs(showing.timeline_out - clip.timeline_in) < 0.002
          && time >= clip.timeline_in - clip.transition.seconds);
        if (incoming) this.drawTransition(track, incoming, time);
      }
      this.drawCards(time);
      this.drawCaptions(time);
    }

    // A title card's words, where the render puts them, fading in and out as they do there.
    drawCards(time) {
      const style = this.described.cards;
      const event = style?.events.find((item) => time >= item.start && time < item.end);
      if (!event) return;
      const { ctx } = this;
      const font = this.described.captions.font;
      const shown = clamp(Math.min((time - event.start) / event.fade, (event.end - time) / event.fade), 0, 1);
      ctx.save();
      ctx.globalAlpha = shown;
      ctx.textAlign = "center";
      ctx.textBaseline = "middle";
      for (const row of event.rows) {
        ctx.font = `bold ${row.size}px "${font}", ${FONTS}`;
        const shadow = Math.max(1, Math.floor(row.size / 24));
        ctx.fillStyle = "rgba(0,0,0,.5)";
        ctx.fillText(row.text, this.described.width / 2 + shadow, row.y + shadow);
        ctx.fillStyle = "#ffffff";
        ctx.fillText(row.text, this.described.width / 2, row.y);
      }
      ctx.restore();
    }

    drawTransition(track, clip, time) {
      const { ctx } = this;
      const { kind, seconds } = clip.transition;
      const done = clamp((time - (clip.timeline_in - seconds)) / seconds, 0, 1);
      const [x, y, w, h] = clip.box;
      ctx.save();
      if (kind === "dissolve") {
        this.drawClip(track, clip, time, done);
      } else if (kind === "wipe") {
        const direction = clip.transition.direction || "left";
        ctx.beginPath();
        if (direction === "left") ctx.rect(x + w * (1 - done), y, w * done, h);
        else if (direction === "right") ctx.rect(x, y, w * done, h);
        else if (direction === "up") ctx.rect(x, y + h * (1 - done), w, h * done);
        else ctx.rect(x, y, w, h * done);
        ctx.clip();
        this.drawClip(track, clip, time);
      } else {
        const through = clip.transition.through || "black";
        ctx.fillStyle = through;
        ctx.globalAlpha = done < 0.5 ? done * 2 : 1;
        ctx.fillRect(x, y, w, h);
        if (done >= 0.5) this.drawClip(track, clip, time, done * 2 - 1);
      }
      ctx.restore();
    }

    drawClip(track, clip, time, alpha = 1) {
      const { ctx } = this;
      const [x, y, w, h] = clip.box;
      const slot = this.slots.get(`${track.id}/${clip.id}`);
      const element = slot?.picture;
      const vw = element?.videoWidth || element?.naturalWidth, vh = element?.videoHeight || element?.naturalHeight;
      if (!element || (!clip.still && element.readyState < 2) || !vw) {
        // Not there yet: its copy is still being made, or still opening.
        ctx.save();
        ctx.globalAlpha = alpha;
        ctx.fillStyle = "#18181b";
        ctx.fillRect(x, y, w, h);
        ctx.restore();
        return;
      }
      // Scaled to cover its box, closer when zoomed, and the rest cropped where the framing says.
      const zoom = clip.crop?.zoom || 1;
      const cover = Math.max(w / vw, h / vh) * zoom;
      const sw = w / cover, sh = h / cover;
      let centre = 0.5;
      const steps = clip.crop?.steps || [];
      const into = time - pictureStart(clip);
      for (const [at, value] of steps) if (at <= into + 1e-6) centre = value;
      const axis = clip.crop?.axis;
      const cross = clip.crop?.cross ?? 0.5;
      const sx = clamp((axis === "x" ? centre : cross) * vw - sw / 2, 0, vw - sw);
      const sy = clamp((axis === "y" ? centre : cross) * vh - sh / 2, 0, vh - sh);
      ctx.save();
      ctx.globalAlpha = alpha;
      if (clip.still) {
        // A photo moves over the placed picture, the way the render's zoompan does: closer by the
        // zoom, the view's left edge `across` of the way along the room the zoom leaves.
        const [z, across] = motionAt(clip, (time - pictureStart(clip)) / Math.max(1e-3, clip.timeline_out - pictureStart(clip)));
        ctx.save();
        ctx.beginPath();
        ctx.rect(x, y, w, h);
        ctx.clip();
        ctx.translate(x - (w * z - w) * across, y - (h * z - h) / 2);
        ctx.scale(z, z);
        ctx.translate(-x, -y);
      }
      const color = clip.color;
      let look = color ? `brightness(${1 + color.brightness}) contrast(${color.contrast}) saturate(${color.saturation})` : "";
      // A title card's background: its frame blurred and darkened, darkened only, or a plain colour.
      const card = clip.card, cardStyle = this.described.cards;
      if (card?.background === "blur") {
        look += ` blur(${Math.max(2, Math.round(Math.min(w, h) * cardStyle.blur_share))}px) brightness(${1 + cardStyle.blur_dim})`;
      } else if (card?.background === "picture") {
        look += ` brightness(${1 + cardStyle.picture_dim})`;
      }
      if (card?.background === "colour") {
        ctx.fillStyle = card.colour;
        ctx.fillRect(x, y, w, h);
      } else if (clip.crop?.whole) {
        // All of it, fitted inside, over a blurred and dimmed copy covering the box — as the render does.
        const base = Math.max(w / vw, h / vh), fit = Math.min(w / vw, h / vh);
        const bw = w / base, bh = h / base;
        ctx.filter = `${look} blur(${Math.max(4, Math.round(Math.min(w, h) / 30))}px) brightness(${1 + WHOLE_DIM})`;
        ctx.drawImage(element, (vw - bw) / 2, (vh - bh) / 2, bw, bh, x, y, w, h);
        ctx.filter = look || "none";
        const fw = vw * fit, fh = vh * fit;
        ctx.drawImage(element, 0, 0, vw, vh, x + (w - fw) / 2, y + (h - fh) / 2, fw, fh);
      } else {
        if (look) ctx.filter = look;
        ctx.drawImage(element, sx, sy, sw, sh, x, y, w, h);
      }
      ctx.filter = "none";
      if (clip.still) ctx.restore();
      if (color?.temperature) {
        // Warmer below daylight, cooler above it: a tint, where a render shifts the white point.
        const warm = color.temperature < 6500;
        const strength = warm ? (6500 - color.temperature) / 6500 * 0.5 : (color.temperature - 6500) / 30000;
        ctx.globalCompositeOperation = "soft-light";
        ctx.globalAlpha = alpha * clamp(strength, 0, 0.4);
        ctx.fillStyle = warm ? "#ff8a2a" : "#3c8cff";
        ctx.fillRect(x, y, w, h);
        ctx.globalCompositeOperation = "source-over";
      }
      // Fades go to black, counted from the first frame the clip shows.
      const start = pictureStart(clip);
      let shown = 1;
      if (clip.video_fade_in > 0) shown *= clamp((time - start) / clip.video_fade_in, 0, 1);
      if (clip.video_fade_out > 0) shown *= clamp((clip.timeline_out - time) / clip.video_fade_out, 0, 1);
      if (shown < 1) {
        ctx.globalAlpha = alpha * (1 - shown);
        ctx.fillStyle = "#000";
        ctx.fillRect(x, y, w, h);
      }
      ctx.restore();
    }

    drawCaptions(time) {
      const captions = this.described.captions;
      const event = captions.events.find((item) => time >= item.start && time < item.end);
      if (!event) return;
      const { ctx } = this;
      const { width, height } = this.described;
      const font = (size) => `bold ${size}px "${captions.font}", ${FONTS}`;
      const lit = event.colour || "#ffffff";
      // Bottom up: the second language under the first, the block's foot on the margin.
      let foot = height - captions.margin_v;
      const rows = [];
      for (const line of [...event.secondary].reverse()) {
        rows.push({ size: captions.secondary_size, words: [[line, null, null]], foot });
        foot -= captions.secondary_size * captions.line_height;
      }
      const lines = event.lines.map((words, index) => (index === 0 && event.prefix ? [[event.prefix, null, null], ...words] : words));
      for (const words of [...lines].reverse()) {
        rows.push({ size: captions.font_size, words, foot });
        foot -= captions.font_size * captions.line_height;
      }
      ctx.save();
      ctx.lineJoin = "round";
      ctx.textBaseline = "alphabetic";
      for (const row of rows) {
        ctx.font = font(row.size);
        const widths = row.words.map(([text]) => ctx.measureText(text).width);
        let left = width / 2 - widths.reduce((a, b) => a + b, 0) / 2;
        const baseline = row.foot - row.size * 0.22;
        row.words.forEach(([text, start], index) => {
          const spoken = start === null || time >= start;
          const paint = (dx, dy, fill, stroke) => {
            if (captions.outline) {
              ctx.strokeStyle = stroke;
              ctx.lineWidth = captions.outline * 2;
              ctx.strokeText(text, left + dx, baseline + dy);
            }
            ctx.fillStyle = fill;
            ctx.fillText(text, left + dx, baseline + dy);
          };
          if (captions.shadow) paint(captions.shadow, captions.shadow, "rgba(0,0,0,.5)", "rgba(0,0,0,.5)");
          paint(0, 0, spoken ? lit : captions.unspoken, "#000");
          left += widths[index];
        });
      }
      ctx.restore();
    }
  }

  window.Player = { create: (stage, onState) => new Player(stage, onState) };
})();
