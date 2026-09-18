import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

// Keep the preview as a normal LiteGraph DOM widget.  The widget contributes an
// absolute height to node.computeSize(); it never adds its height to the node's
// current height.  This is the important invariant that prevents the Combine
// node from growing after every execute/configure/interaction.
const PREVIEW_MIN_HEIGHT = 72;
const PREVIEW_MAX_HEIGHT = 720;
const PREVIEW_HORIZONTAL_MARGIN = 20;
const DEFAULT_ASPECT = 16 / 9;
const MIN_NODE_WIDTH = 320;

function previewUrl(preview) {
    const params = new URLSearchParams();
    params.set("filename", preview?.filename || "");
    params.set("subfolder", preview?.subfolder || "");
    params.set("type", preview?.type || "output");
    // A new encode may reuse a filename. Avoid a stale browser media cache.
    params.set("aetherscale_ts", String(Date.now()));
    return api.apiURL(`/view?${params.toString()}`);
}

function clamp(value, min, max) {
    return Math.max(min, Math.min(max, value));
}

function validAspect(value) {
    const aspect = Number(value);
    return Number.isFinite(aspect) && aspect > 0 ? aspect : DEFAULT_ASPECT;
}

function previewHeightForWidth(node, width, aspectOverride = null) {
    const nodeWidth = Math.max(MIN_NODE_WIDTH, Number(width || node?.size?.[0] || 420));
    const contentWidth = Math.max(1, nodeWidth - PREVIEW_HORIZONTAL_MARGIN);
    const aspect = validAspect(aspectOverride ?? node?.__aetherscalePreviewAspect);
    // The +10 mirrors LiteGraph's widget spacing and keeps controls away from the
    // node edge.  Height is derived from width, never from the current node height.
    return Math.round(clamp(contentWidth / aspect + 10, PREVIEW_MIN_HEIGHT, PREVIEW_MAX_HEIGHT));
}

function markDirty(node) {
    try { node?.setDirtyCanvas?.(true, true); } catch (_) {}
    try { node?.graph?.setDirtyCanvas?.(true, true); } catch (_) {}
}

function fitHeight(node) {
    if (!node || node.__aetherscalePreviewFitting) return;
    try {
        node.__aetherscalePreviewFitting = true;
        const width = Math.max(MIN_NODE_WIDTH, Number(node.size?.[0] || 420));
        // Follow the same layout principle as VHS: let LiteGraph recompute the
        // complete node height from all widgets, including the preview widget.
        // Never use "current height + preview height".
        const computed = node.computeSize?.([width, Number(node.size?.[1] || 0)]);
        const targetHeight = Number(computed?.[1]);
        if (Number.isFinite(targetHeight) && targetHeight > 0) {
            const currentHeight = Number(node.size?.[1] || 0);
            if (Math.abs(currentHeight - targetHeight) > 1) {
                node.setSize?.([width, targetHeight]);
            }
        }
        markDirty(node);
    } catch (error) {
        console.warn("[AetherScale] Video preview fit skipped:", error);
    } finally {
        node.__aetherscalePreviewFitting = false;
    }
}

function scheduleFit(node) {
    if (!node || node.__aetherscalePreviewFitPending) return;
    node.__aetherscalePreviewFitPending = true;
    requestAnimationFrame(() => {
        node.__aetherscalePreviewFitPending = false;
        fitHeight(node);
    });
}

function clearMedia(video) {
    if (!video) return;
    try {
        video.pause?.();
        video.removeAttribute?.("src");
        video.load?.();
    } catch (_) {}
}

function removePreview(node) {
    try {
        const widget = node?.__aetherscaleVideoPreviewWidget;
        if (widget) {
            try { widget.onRemove?.(); } catch (_) {}
            const index = node.widgets?.indexOf(widget) ?? -1;
            if (index >= 0) node.widgets.splice(index, 1);
        }
        clearMedia(node?.__aetherscaleVideoPreviewElement);
        try { node?.__aetherscaleVideoPreviewWrapper?.remove?.(); } catch (_) {}
        if (node) {
            node.__aetherscaleVideoPreviewWidget = null;
            node.__aetherscaleVideoPreviewElement = null;
            node.__aetherscaleVideoPreviewWrapper = null;
            node.__aetherscalePreviewAspect = null;
            node.__aetherscalePreviewHasSource = false;
        }
    } catch (error) {
        console.warn("[AetherScale] Video preview cleanup skipped:", error);
    }
}

function ensurePreviewWidget(node) {
    if (!node || typeof node.addDOMWidget !== "function") return null;
    if (node.__aetherscaleVideoPreviewWidget) return node.__aetherscaleVideoPreviewWidget;

    const wrapper = document.createElement("div");
    wrapper.className = "aetherscale-video-preview";
    wrapper.hidden = true;
    Object.assign(wrapper.style, {
        width: "100%",
        height: "100%",
        boxSizing: "border-box",
        alignItems: "center",
        justifyContent: "center",
        overflow: "hidden",
        background: "#000",
        borderRadius: "6px",
    });

    const video = document.createElement("video");
    video.controls = true;
    video.loop = true;
    video.preload = "metadata";
    video.playsInline = true;
    Object.assign(video.style, {
        width: "100%",
        height: "100%",
        maxWidth: "100%",
        maxHeight: "100%",
        objectFit: "contain",
        display: "block",
        background: "#000",
    });
    wrapper.appendChild(video);

    const widget = node.addDOMWidget("aetherscale_video_preview", "preview", wrapper, {
        serialize: false,
        hideOnZoom: false,
        getValue: () => null,
        setValue: () => {},
    });
    widget.serialize = false;
    widget.aspectRatio = null;
    widget.computeSize = function (width) {
        if (!wrapper.hidden && node.__aetherscalePreviewHasSource) {
            const aspect = validAspect(this.aspectRatio ?? node.__aetherscalePreviewAspect);
            const height = previewHeightForWidth(node, width, aspect);
            this.computedHeight = height;
            return [Math.max(1, Number(width || node.size?.[0] || 420)), height];
        }
        // A preview with no source contributes no visible height to the node instead
        // of reserving a permanent empty rectangle.
        return [Math.max(1, Number(width || node.size?.[0] || 420)), -4];
    };

    video.addEventListener("loadedmetadata", () => {
        if (video.videoWidth > 0 && video.videoHeight > 0) {
            const aspect = video.videoWidth / video.videoHeight;
            widget.aspectRatio = aspect;
            node.__aetherscalePreviewAspect = aspect;
        }
        node.__aetherscalePreviewHasSource = true;
        wrapper.hidden = false;
        wrapper.style.display = "flex";
        scheduleFit(node);
    });

    video.addEventListener("error", () => {
        node.__aetherscalePreviewHasSource = false;
        wrapper.hidden = true;
        scheduleFit(node);
    });

    node.__aetherscaleVideoPreviewWidget = widget;
    node.__aetherscaleVideoPreviewElement = video;
    node.__aetherscaleVideoPreviewWrapper = wrapper;
    node.__aetherscalePreviewAspect = DEFAULT_ASPECT;
    node.__aetherscalePreviewHasSource = false;
    return widget;
}

function updatePreview(node, preview) {
    try {
        if (!preview?.filename) return;
        const widget = ensurePreviewWidget(node);
        if (!widget) return;

        const wrapper = node.__aetherscaleVideoPreviewWrapper;
        const video = node.__aetherscaleVideoPreviewElement;
        node.__aetherscalePreviewHasSource = true;
        wrapper.hidden = false;
        wrapper.style.display = "flex";
        video.src = previewUrl(preview);
        video.load?.();

        // Keep the previous aspect while metadata for the new file arrives. This
        // avoids a temporary collapse/re-expand and, importantly, never changes
        // the node by an additive amount.
        scheduleFit(node);
    } catch (error) {
        console.error("[AetherScale] Video preview update disabled for this node:", error);
    }
}

function installStableLayoutHooks(nodeType) {
    const originalCreated = nodeType.prototype.onNodeCreated;
    nodeType.prototype.onNodeCreated = function () {
        const result = originalCreated?.apply(this, arguments);
        try {
            ensurePreviewWidget(this);
            this.__aetherscalePreviewLastWidth = Number(this.size?.[0] || 0);
            // A saved workflow may contain the runaway height produced by older
            // builds. Recompute once after LiteGraph finishes creating the node.
            requestAnimationFrame(() => requestAnimationFrame(() => scheduleFit(this)));
        } catch (error) {
            console.warn("[AetherScale] Preview creation hook skipped:", error);
        }
        return result;
    };

    const originalConfigured = nodeType.prototype.onConfigure;
    nodeType.prototype.onConfigure = function () {
        const result = originalConfigured?.apply(this, arguments);
        try {
            ensurePreviewWidget(this);
            // Heal gigantic saved node heights from v0.7.3-v0.9.1. This computes
            // the absolute natural height; it does not subtract a guessed amount.
            requestAnimationFrame(() => requestAnimationFrame(() => scheduleFit(this)));
        } catch (error) {
            console.warn("[AetherScale] Preview configure hook skipped:", error);
        }
        return result;
    };

    const originalExecuted = nodeType.prototype.onExecuted;
    nodeType.prototype.onExecuted = function (message) {
        const result = originalExecuted?.apply(this, arguments);
        try {
            const preview = message?.gifs?.[0];
            if (preview) updatePreview(this, preview);
        } catch (error) {
            console.warn("[AetherScale] Preview execution hook skipped:", error);
        }
        return result;
    };

    const originalResize = nodeType.prototype.onResize;
    nodeType.prototype.onResize = function () {
        const result = originalResize?.apply(this, arguments);
        try {
            const width = Number(this.size?.[0] || 0);
            const lastWidth = Number(this.__aetherscalePreviewLastWidth || 0);
            this.__aetherscalePreviewLastWidth = width;
            // Only a width change changes an aspect-ratio-driven preview height.
            // Vertical user resizing must not start a resize feedback loop.
            if (this.__aetherscalePreviewHasSource && Math.abs(width - lastWidth) > 0.5) {
                scheduleFit(this);
            }
        } catch (_) {}
        return result;
    };

    const originalRemoved = nodeType.prototype.onRemoved;
    nodeType.prototype.onRemoved = function () {
        try { removePreview(this); } catch (_) {}
        return originalRemoved?.apply(this, arguments);
    };
}

try {
    app.registerExtension({
        name: "AetherScale.VideoCombinePreview",
        async beforeRegisterNodeDef(nodeType, nodeData) {
            try {
                if (nodeData?.name !== "AetherScaleVideoCombine") return;
                installStableLayoutHooks(nodeType);
            } catch (error) {
                console.error("[AetherScale] Video preview registration skipped:", error);
            }
        },
    });
} catch (error) {
    // An optional preview must never prevent ComfyUI from loading its frontend.
    console.error("[AetherScale] Frontend preview extension disabled:", error);
}
