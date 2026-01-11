// VolumetricFogPass (Unity 6 / URP RenderGraph ONLY)
// - No Compatibility Mode (no OnCameraSetup / Execute overrides)
// - Uses RecordRenderGraph + UniversalResourceData activeColor/depth
//
// REQUIREMENT:
// Your FogVolume overload must accept RasterCommandBuffer:
//
// internal void DrawVolume(Camera camera, in RenderTextureDescriptor cameraTargetDesc, bool isSceneViewCamera,
//                          RasterCommandBuffer cmd, Shader shader, List<NativeLight> lights, int maxLights)

using UnityEngine;
using UnityEngine.Rendering;
using UnityEngine.Rendering.Universal;
using UnityEngine.Rendering.RenderGraphModule;

using Unity.Collections;

using System.Collections.Generic;

namespace Sinnwrig.FogVolumes
{
    public class VolumetricFogPass : ScriptableRenderPass
    {
        // Bundles the integer ID with an RtID (for global shader properties)
        readonly struct RTPair
        {
            public readonly int propertyId;
            public readonly RenderTargetIdentifier identifier;

            public RTPair(string propertyName)
            {
                propertyId = Shader.PropertyToID(propertyName);
                identifier = new RenderTargetIdentifier(propertyId);
            }

            public static implicit operator int(RTPair a) => a.propertyId;
            public static implicit operator RenderTargetIdentifier(RTPair a) => a.identifier;
        }

        // --------------------------------------------------------------------------
        // Globals (shader property IDs)
        // --------------------------------------------------------------------------

        private static readonly RTPair halfDepth = new RTPair("_HalfDepthTarget");
        private static readonly RTPair quarterDepth = new RTPair("_QuarterDepthTarget");

        private static readonly RTPair volumeFog = new RTPair("_VolumeFogTexture");
        private static readonly RTPair halfVolumeFog = new RTPair("_HalfVolumeFogTexture");
        private static readonly RTPair quarterVolumeFog = new RTPair("_QuarterVolumeFogTexture");

        private static readonly RTPair temporalTarget = new RTPair("_TemporalTarget");

        // Materials / shaders
        private static Material bilateralBlur;
        private static Shader fogShader;
        private static Material blitAdd;
        private static Material reprojection;

        private readonly VolumetricFogFeature feature;

        // --------------------------------------------------------------------------
        // Volumes + culling
        // --------------------------------------------------------------------------

        private static readonly HashSet<FogVolume> activeVolumes = new();
        public static void AddVolume(FogVolume volume) => activeVolumes.Add(volume);
        public static void RemoveVolume(FogVolume volume) => activeVolumes.Remove(volume);
        public static IEnumerable<FogVolume> ActiveVolumes => activeVolumes;

        private static readonly Plane[] cullingPlanes = new Plane[6];

        // --------------------------------------------------------------------------
        // Temporal (kept as you had it; uses persistent RenderTexture)
        // NOTE: This is not URP historyManager-based temporal. It works, but isn't the "modern" way.
        // --------------------------------------------------------------------------

        private static readonly GlobalKeyword temporalKeyword = GlobalKeyword.Create("TEMPORAL_RENDERING_ENABLED");
        private int TemporalKernelSize => System.Math.Max(2, feature.temporalResolution);
        private int temporalPassIndex;

        private RenderTexture temporalBuffer;
        private static System.Random random = new();

        // --------------------------------------------------------------------------
        // Resolution logic
        // --------------------------------------------------------------------------

        private VolumetricResolution Resolution
        {
            get
            {
                // Temporal reprojection will force full-res rendering
                if (feature.resolution != VolumetricResolution.Full && feature.temporalRendering)
                    return VolumetricResolution.Full;

                return feature.resolution;
            }
        }

        // --------------------------------------------------------------------------
        // Constructor
        // --------------------------------------------------------------------------

        public VolumetricFogPass(VolumetricFogFeature feature, Shader blur, Shader fog, Shader add, Shader reproj)
        {
            this.feature = feature;

            fogShader = fog;

            if (bilateralBlur == null || bilateralBlur.shader != blur)
                bilateralBlur = CoreUtils.CreateEngineMaterial(blur);

            if (blitAdd == null || blitAdd.shader != add)
                blitAdd = CoreUtils.CreateEngineMaterial(add);

            if (reprojection == null || reprojection.shader != reproj)
                reprojection = CoreUtils.CreateEngineMaterial(reproj);
        }

        // --------------------------------------------------------------------------
        // RenderGraph plumbing
        // --------------------------------------------------------------------------

        private class PassData
        {
            public VolumetricFogPass self;

            public UniversalCameraData cameraData;
            public UniversalLightData lightData;

            public TextureHandle cameraColor;
            public TextureHandle cameraDepth;

            public TextureHandle volumeFog;
            public TextureHandle halfFog;
            public TextureHandle quarterFog;

            public TextureHandle halfDepth;
            public TextureHandle quarterDepth;

            public TextureHandle blurTempFull;
            public TextureHandle blurTempHalf;
            public TextureHandle blurTempQuarter;

            public TextureHandle copyColor;

            public TextureHandle temporalTarget;
        }

        public override void RecordRenderGraph(RenderGraph renderGraph, ContextContainer frameData)
        {
            var cameraData = frameData.Get<UniversalCameraData>();
            var lightData = frameData.Get<UniversalLightData>();
            var resourceData = frameData.Get<UniversalResourceData>();

            // CPU cull volumes up-front
            var fogVolumes = SetupVolumes(cameraData.camera);
            if (fogVolumes.Count == 0)
                return;

            var camDesc = cameraData.cameraTargetDescriptor;
            int w = camDesc.width;
            int h = camDesc.height;

            // Active camera targets (RenderGraph path)
            var cameraColor = resourceData.activeColorTexture;
            var cameraDepth = resourceData.activeDepthTexture;

            // --- Create fog + depth intermediates ---

            // Full-res fog buffer always exists (even if half/quarter: we upsample into it)
            var volumeFogTex = renderGraph.CreateTexture(new TextureDesc(w, h)
            {
                name = "_VolumeFogTexture",
                colorFormat = UnityEngine.Experimental.Rendering.GraphicsFormat.R16G16B16A16_SFloat,
                clearBuffer = true,
                clearColor = Color.clear,
                filterMode = FilterMode.Point
            });

            TextureHandle halfFogTex = default;
            TextureHandle quarterFogTex = default;
            TextureHandle halfDepthTex = default;
            TextureHandle quarterDepthTex = default;

            if (Resolution == VolumetricResolution.Half || Resolution == VolumetricResolution.Quarter)
            {
                halfDepthTex = renderGraph.CreateTexture(new TextureDesc(Mathf.Max(1, w / 2), Mathf.Max(1, h / 2))
                {
                    name = "_HalfDepthTarget",
                    colorFormat = UnityEngine.Experimental.Rendering.GraphicsFormat.R32_SFloat,
                    clearBuffer = false,
                    filterMode = FilterMode.Point
                });
            }

            if (Resolution == VolumetricResolution.Half)
            {
                halfFogTex = renderGraph.CreateTexture(new TextureDesc(Mathf.Max(1, w / 2), Mathf.Max(1, h / 2))
                {
                    name = "_HalfVolumeFogTexture",
                    colorFormat = UnityEngine.Experimental.Rendering.GraphicsFormat.R16G16B16A16_SFloat,
                    clearBuffer = true,
                    clearColor = Color.clear,
                    filterMode = FilterMode.Bilinear
                });
            }

            if (Resolution == VolumetricResolution.Quarter)
            {
                quarterFogTex = renderGraph.CreateTexture(new TextureDesc(Mathf.Max(1, w / 4), Mathf.Max(1, h / 4))
                {
                    name = "_QuarterVolumeFogTexture",
                    colorFormat = UnityEngine.Experimental.Rendering.GraphicsFormat.R16G16B16A16_SFloat,
                    clearBuffer = true,
                    clearColor = Color.clear,
                    filterMode = FilterMode.Bilinear
                });

                quarterDepthTex = renderGraph.CreateTexture(new TextureDesc(Mathf.Max(1, w / 4), Mathf.Max(1, h / 4))
                {
                    name = "_QuarterDepthTarget",
                    colorFormat = UnityEngine.Experimental.Rendering.GraphicsFormat.R32_SFloat,
                    clearBuffer = false,
                    filterMode = FilterMode.Point
                });
            }

            // --- Blur temps (avoid cmd.GetTemporaryRT in RG) ---
            var blurTempFull = renderGraph.CreateTexture(new TextureDesc(w, h)
            {
                name = "_FogBlurTemp_Full",
                colorFormat = UnityEngine.Experimental.Rendering.GraphicsFormat.R16G16B16A16_SFloat,
                clearBuffer = false,
                filterMode = FilterMode.Bilinear
            });

            TextureHandle blurTempHalf = default;
            TextureHandle blurTempQuarter = default;

            if (Resolution == VolumetricResolution.Half)
            {
                blurTempHalf = renderGraph.CreateTexture(new TextureDesc(Mathf.Max(1, w / 2), Mathf.Max(1, h / 2))
                {
                    name = "_FogBlurTemp_Half",
                    colorFormat = UnityEngine.Experimental.Rendering.GraphicsFormat.R16G16B16A16_SFloat,
                    clearBuffer = false,
                    filterMode = FilterMode.Bilinear
                });
            }

            if (Resolution == VolumetricResolution.Quarter)
            {
                blurTempQuarter = renderGraph.CreateTexture(new TextureDesc(Mathf.Max(1, w / 4), Mathf.Max(1, h / 4))
                {
                    name = "_FogBlurTemp_Quarter",
                    colorFormat = UnityEngine.Experimental.Rendering.GraphicsFormat.R16G16B16A16_SFloat,
                    clearBuffer = false,
                    filterMode = FilterMode.Bilinear
                });
            }

            // --- Camera color copy for blending (avoid read/write same texture) ---
            var copyColor = renderGraph.CreateTexture(new TextureDesc(w, h)
            {
                name = "_FogSceneColorCopy",
                colorFormat = resourceData.activeColorTexture.GetDescriptor(renderGraph).colorFormat,
                clearBuffer = false,
                filterMode = FilterMode.Point
            });

            // --- Temporal target (low-res) ---
            TextureHandle temporalTargetTex = default;
            if (feature.temporalRendering)
            {
                temporalTargetTex = renderGraph.CreateTexture(new TextureDesc(Mathf.Max(1, w / TemporalKernelSize), Mathf.Max(1, h / TemporalKernelSize))
                {
                    name = "_TemporalTarget",
                    colorFormat = UnityEngine.Experimental.Rendering.GraphicsFormat.R16G16B16A16_SFloat,
                    clearBuffer = true,
                    clearColor = Color.clear,
                    filterMode = FilterMode.Point
                });
            }

            using var builder = renderGraph.AddUnsafePass<PassData>("Volumetric Fog", out var passData);


            passData.self = this;
            passData.cameraData = cameraData;
            passData.lightData = lightData;

            passData.cameraColor = cameraColor;
            passData.cameraDepth = cameraDepth;

            passData.volumeFog = volumeFogTex;
            passData.halfFog = halfFogTex;
            passData.quarterFog = quarterFogTex;

            passData.halfDepth = halfDepthTex;
            passData.quarterDepth = quarterDepthTex;

            passData.blurTempFull = blurTempFull;
            passData.blurTempHalf = blurTempHalf;
            passData.blurTempQuarter = blurTempQuarter;

            passData.copyColor = copyColor;

            passData.temporalTarget = temporalTargetTex;

            // Declare usage
            builder.UseTexture(cameraColor, AccessFlags.ReadWrite);
            builder.UseTexture(cameraDepth, AccessFlags.Read);

            builder.UseTexture(volumeFogTex, AccessFlags.ReadWrite);
            builder.UseTexture(blurTempFull, AccessFlags.ReadWrite);

            builder.UseTexture(copyColor, AccessFlags.ReadWrite);

            if (halfFogTex.IsValid()) builder.UseTexture(halfFogTex, AccessFlags.ReadWrite);
            if (quarterFogTex.IsValid()) builder.UseTexture(quarterFogTex, AccessFlags.ReadWrite);
            if (halfDepthTex.IsValid()) builder.UseTexture(halfDepthTex, AccessFlags.ReadWrite);
            if (quarterDepthTex.IsValid()) builder.UseTexture(quarterDepthTex, AccessFlags.ReadWrite);
            if (blurTempHalf.IsValid()) builder.UseTexture(blurTempHalf, AccessFlags.ReadWrite);
            if (blurTempQuarter.IsValid()) builder.UseTexture(blurTempQuarter, AccessFlags.ReadWrite);
            if (temporalTargetTex.IsValid()) builder.UseTexture(temporalTargetTex, AccessFlags.ReadWrite);

            builder.SetRenderFunc((PassData data, UnsafeGraphContext ctx) =>
            {
                var self = data.self;
                var cmd = ctx.cmd;

                // Bind shader globals (by ID, as your shaders expect)
                ctx.cmd.SetGlobalTexture(volumeFog, data.volumeFog);
                if (data.halfFog.IsValid()) ctx.cmd.SetGlobalTexture(halfVolumeFog, data.halfFog);
                if (data.quarterFog.IsValid()) ctx.cmd.SetGlobalTexture(quarterVolumeFog, data.quarterFog);
                if (data.halfDepth.IsValid()) ctx.cmd.SetGlobalTexture(halfDepth, data.halfDepth);
                if (data.quarterDepth.IsValid()) ctx.cmd.SetGlobalTexture(quarterDepth, data.quarterDepth);
                if (data.temporalTarget.IsValid()) ctx.cmd.SetGlobalTexture(temporalTarget, data.temporalTarget);

                // 1) Downsample depth into half/quarter depth RTs if needed
                self.DownsampleDepthBuffer_RG(ctx.cmd, data.cameraDepth, data.halfDepth, data.quarterDepth);

                // 2) Choose fog draw target and clear it
                var descr = data.cameraData.cameraTargetDescriptor;
                self.SetupFogRenderTarget_RG(cmd, descr.width, descr.height, data.volumeFog, data.halfFog, data.quarterFog, data.temporalTarget);


                // 3) Build light list
                var lights = self.SetupLights_RG(data.cameraData, data.lightData);

                int maxLights = data.lightData.additionalLightsCount;
                var desc = data.cameraData.cameraTargetDescriptor;
                bool isSceneView = data.cameraData.isSceneViewCamera;

                // 4) Draw volumes
                for (int i = 0; i < fogVolumes.Count; i++)
                {
                    fogVolumes[i].DrawVolume(data.cameraData.camera, in desc, isSceneView, ctx.cmd, fogShader, lights, maxLights);
                }

                // 5) Temporal reprojection (optional)
                if (feature.temporalRendering)
                {
                    self.ReprojectBuffer_RG(ctx.cmd, in desc, data.temporalTarget, data.volumeFog, isSceneView);
                }

                // 6) Blur + upscale
                self.BilateralBlur_RG(
                    ctx.cmd,
                    width: desc.width,
                    height: desc.height,
                    cameraDepth: data.cameraDepth,
                    fullFog: data.volumeFog,
                    halfFog: data.halfFog,
                    quarterFog: data.quarterFog,
                    halfDepthTex: data.halfDepth,
                    quarterDepthTex: data.quarterDepth,
                    blurTempFull: data.blurTempFull,
                    blurTempHalf: data.blurTempHalf,
                    blurTempQuarter: data.blurTempQuarter
                );

                // 7) Blend into camera color
                ctx.cmd.CopyTexture(data.cameraColor, data.copyColor);

                ctx.cmd.SetGlobalTexture("_BlitSource", data.copyColor);
                ctx.cmd.SetGlobalTexture("_BlitAdd", data.volumeFog);

                TargetBlit(ctx.cmd, data.cameraColor, blitAdd, 0);
            });
        }

        // --------------------------------------------------------------------------
        // Helpers
        // --------------------------------------------------------------------------

        private List<FogVolume> SetupVolumes(Camera camera)
        {
            GeometryUtility.CalculateFrustumPlanes(camera, cullingPlanes);
            Vector3 camPos = camera.transform.position;

            List<FogVolume> fogVolumes = new();
            foreach (FogVolume volume in activeVolumes)
            {
                if (!volume.CullVolume(camPos, cullingPlanes))
                    fogVolumes.Add(volume);
            }
            return fogVolumes;
        }

        private bool CullSphere(Vector3 pos, float radius)
        {
            for (int i = 0; i < cullingPlanes.Length; i++)
            {
                float distance = cullingPlanes[i].GetDistanceToPoint(pos);
                if (distance < 0 && Mathf.Abs(distance) > radius)
                    return true;
            }
            return false;
        }

        private List<NativeLight> SetupLights_RG(UniversalCameraData cameraData, UniversalLightData lightData)
        {
            NativeArray<VisibleLight> visibleLights = lightData.visibleLights;

            List<NativeLight> initializedLights = new();

            Vector3 cameraPosition = cameraData.camera.transform.position;

            for (int i = 0; i < visibleLights.Length; i++)
            {
                var visibleLight = visibleLights[i];

                bool isDirectional = visibleLight.lightType == LightType.Directional;
                bool isMain = (i == lightData.mainLightIndex);

                Vector3 position = visibleLight.localToWorldMatrix.GetColumn(3);

                if (!isDirectional && CullSphere(position, visibleLight.range))
                    continue;

                NativeLight light = new()
                {
                    isDirectional = isDirectional,
                    shadowIndex = isMain ? -1 : -1,
                    range = visibleLight.range,
                    layer = visibleLight.light.gameObject.layer,
                    cameraDistance = isDirectional ? 0 : (cameraPosition - position).sqrMagnitude
                };

                UniversalRenderPipeline.InitializeLightConstants_Common(visibleLights, i,
                    out light.position,
                    out light.color,
                    out light.attenuation,
                    out light.spotDirection,
                    out _
                );

                initializedLights.Add(light);
            }

            initializedLights.Sort((a, b) => a.cameraDistance.CompareTo(b.cameraDistance));
            return initializedLights;
        }

        private void DownsampleDepthBuffer_RG(UnsafeCommandBuffer cmd, TextureHandle cameraDepth, TextureHandle halfDepthTex, TextureHandle quarterDepthTex)
        {
            if (Resolution == VolumetricResolution.Half || Resolution == VolumetricResolution.Quarter)
            {
                cmd.SetGlobalTexture("_DownsampleSource", cameraDepth);
                TargetBlit(cmd, halfDepthTex, bilateralBlur, 2);
            }

            if (Resolution == VolumetricResolution.Quarter)
            {
                cmd.SetGlobalTexture("_DownsampleSource", halfDepthTex);
                TargetBlit(cmd, quarterDepthTex, bilateralBlur, 2);
            }
        }

        private void SetupFogRenderTarget_RG(UnsafeCommandBuffer cmd, int width, int height, TextureHandle fullFog, TextureHandle halfFog, TextureHandle quarterFog, TextureHandle temporalTargetTex)
        {
            cmd.SetKeyword(temporalKeyword, feature.temporalRendering);

            if (Resolution == VolumetricResolution.Quarter)
                cmd.SetRenderTarget(quarterFog);
            else if (Resolution == VolumetricResolution.Half)
                cmd.SetRenderTarget(halfFog);
            else if (feature.temporalRendering)
                cmd.SetRenderTarget(temporalTargetTex);
            else
                cmd.SetRenderTarget(fullFog);

            cmd.ClearRenderTarget(true, true, new Color(0, 0, 0, 0));

            if (feature.temporalRendering)
            {
                SetTemporalConstants(cmd);
                // Render size is full-res, but rendered in tiles
                cmd.SetGlobalVector("_TemporalRenderSize", new Vector2(width, height) / TemporalKernelSize);
            }
        }

        private void SetTemporalConstants(UnsafeCommandBuffer cmd)
        {
            temporalPassIndex = (temporalPassIndex + 1) % (TemporalKernelSize * TemporalKernelSize);

            cmd.SetGlobalVector("_TileSize", new Vector2(TemporalKernelSize, TemporalKernelSize));
            cmd.SetGlobalVector("_PassOffset", new Vector2(random.Next(0, TemporalKernelSize), random.Next(0, TemporalKernelSize)));
        }

        private void ReprojectBuffer_RG(UnsafeCommandBuffer cmd, in RenderTextureDescriptor cameraDesc, TextureHandle temporalTargetTex, TextureHandle volumeFogTex, bool isSceneView)
        {
            int width = cameraDesc.width;
            int height = cameraDesc.height;

            // Ensure persistent history RT exists
            var desc = cameraDesc;
            desc.depthBufferBits = 0;
            desc.msaaSamples = 1;
            desc.colorFormat = RenderTextureFormat.ARGBHalf;

            if (temporalBuffer == null || !temporalBuffer.IsCreated() || temporalBuffer.width != width || temporalBuffer.height != height)
            {
                if (temporalBuffer != null && temporalBuffer.IsCreated())
                    temporalBuffer.Release();

                temporalBuffer = new RenderTexture(desc);
                temporalBuffer.Create();
            }

            cmd.SetGlobalTexture("_TemporalBuffer", temporalBuffer);
            cmd.SetGlobalTexture("_TemporalTarget", temporalTargetTex);
            cmd.SetGlobalFloat("_MotionInfluence", isSceneView ? 0f : 1f);

            // Reproject into full-res volume fog buffer
            TargetBlit(cmd, volumeFogTex, reprojection, 0);

            // Copy current fog into persistent history
            cmd.CopyTexture(volumeFogTex, 0, 0, temporalBuffer, 0, 0);
        }

        private void BilateralBlur_RG(
            UnsafeCommandBuffer cmd,
            int width,
            int height,
            TextureHandle cameraDepth,
            TextureHandle fullFog,
            TextureHandle halfFog,
            TextureHandle quarterFog,
            TextureHandle halfDepthTex,
            TextureHandle quarterDepthTex,
            TextureHandle blurTempFull,
            TextureHandle blurTempHalf,
            TextureHandle blurTempQuarter)
        {
            Resolution.SetResolutionKeyword(cmd);

            // Quarter -> blur + upsample to full
            if (Resolution == VolumetricResolution.Quarter)
            {
                if (!feature.disableBlur)
                    BilateralBlur_RG(cmd, quarterFog, quarterDepthTex, blurTempQuarter);

                Upsample_RG(cmd, quarterFog, quarterDepthTex, fullFog);
                return;
            }

            // Half -> blur + upsample to full
            if (Resolution == VolumetricResolution.Half)
            {
                if (!feature.disableBlur)
                    BilateralBlur_RG(cmd, halfFog, halfDepthTex, blurTempHalf);

                Upsample_RG(cmd, halfFog, halfDepthTex, fullFog);
                return;
            }

            // Full -> blur in place (unless disabled)
            if (feature.disableBlur)
                return;

            BilateralBlur_RG(cmd, fullFog, cameraDepth, blurTempFull);
        }

        private void BilateralBlur_RG(UnsafeCommandBuffer cmd, TextureHandle source, TextureHandle depth, TextureHandle blurTemp)
        {
            cmd.SetGlobalTexture("_DepthTexture", depth);

            // Horizontal blur
            cmd.SetGlobalTexture("_BlurSource", source);
            TargetBlit(cmd, blurTemp, bilateralBlur, 0);

            // Vertical blur
            cmd.SetGlobalTexture("_BlurSource", blurTemp);
            TargetBlit(cmd, source, bilateralBlur, 1);
        }

        private void Upsample_RG(UnsafeCommandBuffer cmd, TextureHandle sourceColor, TextureHandle sourceDepth, TextureHandle destination)
        {
            cmd.SetGlobalTexture("_DownsampleColor", sourceColor);
            cmd.SetGlobalTexture("_DownsampleDepth", sourceDepth);

            TargetBlit(cmd, destination, bilateralBlur, 3);
        }

        // Fullscreen quad blit helper
        private static void TargetBlit(UnsafeCommandBuffer cmd, RenderTargetIdentifier destination, Material material, int pass)
        {
            cmd.SetRenderTarget(destination);
            cmd.DrawMesh(MeshUtility.FullscreenQuad, Matrix4x4.identity, material, 0, pass);
        }

        public void Dispose()
        {
            if (temporalBuffer != null && temporalBuffer.IsCreated())
                temporalBuffer.Release();
        }
    }
}
