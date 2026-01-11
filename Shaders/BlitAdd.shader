Shader "Hidden/BlitAdd"
{	

	SubShader
	{
		// Pass 0 - Blit add into result
		Pass
		{
			Cull Off ZWrite Off ZTest Off

			HLSLPROGRAM

			#pragma exclude_renderers d3d11_9x
    		#pragma exclude_renderers d3d9

			#pragma vertex vert
			#pragma fragment blendFrag
			#pragma target 4.0

			#include "/Include/Common.hlsl"
			#include "/Include/Reprojection.hlsl"
	
			struct appdata
			{
				float4 vertex : POSITION;
				float2 uv : TEXCOORD0;
			};


			struct v2f
			{
				float2 uv : TEXCOORD0;
				float4 vertex : SV_POSITION;
			};


			v2f vert(appdata v)
			{
				v2f o;
				o.vertex = CorrectUV(v.vertex);
				o.uv = v.uv;
				return o;
			}


			TEXTURE2D(_BlitSource);
			SAMPLER(sampler_BlitSource);

			TEXTURE2D(_BlitAdd);
			SAMPLER(sampler_BlitAdd);


float4 blendFrag(v2f i) : SV_Target
{
    float3 baseCol = SAMPLE_BASE(_BlitSource, sampler_BlitSource, i.uv).rgb;
    float4 fog     = SAMPLE_BASE(_BlitAdd, sampler_BlitAdd, i.uv);

    // fog.a should be opacity in [0..1]
    float srcFactor = 1.0 - saturate(fog.a);

    float3 outCol = baseCol * srcFactor + fog.rgb;
    //return float4(0.1, 0.0, 0.0, 0.2);

	return float4(outCol, 1.0);
}

			ENDHLSL
		}
	}
}
