"""Sunny portrait retouch for the provided IMG_6224 original.

Preserves pixel dimensions; no generative AI is needed.
Install: pip install opencv-python numpy
Usage: python portrait_grade.py 'IMG_6224(2).jpeg' -o retouched.jpg

The sky/skin/lips/flowers masks are tuned to THIS portrait's composition.
For other images, adjust the normalized mask coordinates below.
"""
import argparse
import cv2
import numpy as np


def ellipse(h, w, x, y, rx, ry, feather=0.45, angle=0):
    """Soft elliptical selection with normalized coordinates."""
    m = np.zeros((h, w), np.float32)
    center = (round(x*w), round(y*h))
    axes = (max(1, round(rx*w)), max(1, round(ry*h)))
    cv2.ellipse(m, center, axes, angle, 0, 360, 1., -1, cv2.LINE_AA)
    blur_px = max(3, round(min(h, w)*min(rx, ry)*feather))
    return cv2.GaussianBlur(m, (0,0), blur_px)


def blend(base, overlay, alpha):
    if isinstance(alpha, np.ndarray) and alpha.ndim == 2:
        alpha = alpha[:, :, None]
    return base*(1-alpha) + overlay*alpha


def luminance(rgb):
    return rgb[:,:,0]*.2126+rgb[:,:,1]*.7152+rgb[:,:,2]*.0722


def process(path, output, intensity=1.0):
    bgr = cv2.imread(path, cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(path)
    h, w = bgr.shape[:2]
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32)/255.
    yy, xx = np.mgrid[:h, :w].astype(np.float32)
    xn, yn = xx/w, yy/h
    S = np.clip(intensity, 0., 1.8)

    # 1) Re-create some sky color in the white/clipped part of the original.
    #    Only include bright low-saturation regions connected to the TOP edge.
    hsv = cv2.cvtColor((rgb*255).astype(np.uint8), cv2.COLOR_RGB2HSV)
    white = (np.min(rgb, axis=2) > .835) & (hsv[:,:,1] < 44) & (yn < .46)
    white = white.astype(np.uint8)
    white = cv2.morphologyEx(white, cv2.MORPH_CLOSE, np.ones((5,5),np.uint8))
    n, labels, stats, _ = cv2.connectedComponentsWithStats(white, 8)
    sky_mask = np.zeros((h,w), np.float32)
    for component in range(1,n):
        # True sky touches image top and is not an isolated cloud/piece of fabric.
        if stats[component, cv2.CC_STAT_TOP] == 0 and stats[component,cv2.CC_STAT_AREA] > w*h*.015:
            sky_mask[labels == component] = 1
    sky_mask = cv2.GaussianBlur(sky_mask,(0,0), 3.2)
    sky_mask *= np.clip((.465-yn)/.045, 0,1) * S

    # A gentle pale-blue sky gradient, much softer near the trees.
    blue_strength = np.clip(.92 - yn*1.15, .18,.9)
    top = np.array([.54,.74,.96],np.float32)
    bottom = np.array([.87,.92,.97],np.float32)
    sky = bottom[None,None,:] * (1-blue_strength[:,:,None]) + top[None,None,:]*blue_strength[:,:,None]
    # Use very soft cloud texture: don't invent hard cloud shapes.
    noise = np.random.default_rng(7).normal(size=(25,15)).astype(np.float32)
    noise = cv2.GaussianBlur(noise,(0,0),1.7)
    noise = cv2.resize(noise,(w,h),interpolation=cv2.INTER_CUBIC)
    noise /= max(1e-6, np.max(np.abs(noise)))
    sky = np.clip(sky + (noise*.065)[:,:,None],0,1)
    rgb = blend(rgb,sky,sky_mask*.87)

    # 2) Sunny overall color grade: warm highlights, restrained deeper shadows.
    # Mild tone curve adds depth without clipping skin.
    rgb = np.clip(rgb,0,1)
    rgb = np.clip(rgb + (.16*S)*(rgb-.5)*(1-2*np.abs(rgb-.5)),0,1)
    y_lum = luminance(rgb)
    highlights = np.clip((y_lum-.48)/.52,0,1)
    shadows = np.clip((.58-y_lum)/.58,0,1)
    rgb += (highlights*(.014*S))[:,:,None]*np.array([1,.46,-.45],np.float32)
    rgb += (shadows*(.010*S))[:,:,None]*np.array([-.35,.00,.9],np.float32)
    rgb = np.clip(rgb,0,1)

    # Adjust global color, keeping the white top and skin from getting neon.
    hsv = cv2.cvtColor((rgb*255).astype(np.uint8),cv2.COLOR_RGB2HSV).astype(np.float32)
    hsv[:,:,1] = np.clip(hsv[:,:,1]*(1+.055*S),0,255)
    rgb = cv2.cvtColor(hsv.astype(np.uint8),cv2.COLOR_HSV2RGB).astype(np.float32)/255.

    # 3) Local portrait masks. These fractions belong to the original framing.
    face = ellipse(h,w, .623,.369,.192,.169, .36, 12)
    shoulders = ellipse(h,w,.52,.616,.46,.13,.42)
    arms = ellipse(h,w,.468,.922,.45,.09,.50)
    skin = np.maximum.reduce([face, shoulders*.70, arms*.68])

    # Gentle texture-preserving smoothing focused on face and shoulders.
    smoothed = cv2.bilateralFilter((rgb*255).astype(np.uint8), d=7,
                                   sigmaColor=23, sigmaSpace=5).astype(np.float32)/255.
    rgb = blend(rgb, smoothed, np.clip(skin*.26*S,0,.48))

    # Spot healing: a small visible cheek/nose-side spot; adjust/remove as needed.
    # Use ORIGINAL IMAGE coordinates expressed as % to remain resolution-independent.
    blemish = np.zeros((h,w),np.uint8)
    for px, py, radius in [(0.716,.347,.0045)]:
        cv2.circle(blemish,(round(px*w),round(py*h)), max(1,round(radius*w)),255,-1)
    healed = cv2.inpaint((rgb*255).astype(np.uint8),blemish,3,cv2.INPAINT_TELEA)
    rgb = blend(rgb,healed.astype(np.float32)/255.,blemish.astype(np.float32)/255.)

    # Subtle jawline lift: local 3–4 px adjustment, not a dramatic face reshape.
    # If you prefer a completely untouched jawline, comment out these lines.
    chin_zone = np.exp(-.5*(((xn-.686)/.095)**2 + ((yn-.462)/.031)**2))
    remap_y = (yy + chin_zone * (3.5*S)).astype(np.float32)
    rgb = cv2.remap(rgb, xx, remap_y, cv2.INTER_LINEAR,
                    borderMode=cv2.BORDER_REFLECT_101)

    # 4) Add lifelike reflector sparkle to face/shoulder/arm.
    # Creamy highlight, NOT a shiny plastic overlay.
    cheek_light = ellipse(h,w,.625,.367,.09,.074,.65,-14)
    brow_light  = ellipse(h,w,.691,.323,.057,.025,.7,-17)
    shoulder_light = ellipse(h,w,.206,.598,.111,.069,.62,22)
    soft_reflector = np.clip(cheek_light*.35 + brow_light*.12 + shoulder_light*.33 + skin*.085,0,1)
    rgb = blend(rgb,np.minimum(1,rgb + np.array([.038,.032,.026],np.float32)),soft_reflector*S)

    # Warm natural cheek blush, shaped to the skin not the background.
    blush = ellipse(h,w,.600,.369,.081,.041,.45,-12)
    blush2 = ellipse(h,w,.740,.367,.027,.023,.50,-6)
    blushmask = np.clip((blush*.15+blush2*.05)*face*S,0,.20)
    rosy = np.clip(rgb + np.array([.23,-.042,-.04],np.float32),0,1)
    rgb = blend(rgb,rosy,blushmask)

    # Soft pinkish-red lips; mask is small and color-dependent to spare teeth/skin.
    lip_region = ellipse(h,w,.684,.413,.052,.018,.25,-14)
    redness = np.maximum(rgb[:,:,0]-rgb[:,:,1],0)
    lipmask = lip_region * np.clip((redness-.015)*7,0,1) * (.35*S)
    pink = np.clip(rgb + np.array([.12,-.017,.005],np.float32),0,1)
    rgb = blend(rgb,pink,lipmask)

    # Under-chin lightening adds separation and reduces shadow/heaviness.
    underchin = ellipse(h,w,.637,.472,.079,.027,.65,-20)
    rgb = blend(rgb, np.minimum(rgb+np.array([.020,.017,.013]),1),underchin*.38*S)

    # 5) Flower/leaf colors: refreshed but subordinate to face.
    bouquet = ellipse(h,w,.31,.72,.34,.25,.17,-17)
    hsv = cv2.cvtColor((np.clip(rgb,0,1)*255).astype(np.uint8),cv2.COLOR_RGB2HSV).astype(np.float32)
    # A smaller enhancement than the portrait, capped to avoid overpowering red.
    target_sat = np.clip(hsv[:,:,1]*(1 + bouquet*.08*S),0,255)
    hsv[:,:,1] = target_sat
    rgb = cv2.cvtColor(hsv.astype(np.uint8),cv2.COLOR_HSV2RGB).astype(np.float32)/255.

    # 6) Details: soft clarity on eyes, hair, jewelry, fabric, and face.
    # Keep blurry foreground bouquet blurry, as it was photographed.
    detail_region = np.clip(face*.76 + ellipse(h,w,.66,.48,.31,.40,.25)*.30,0,1)
    blurred = cv2.GaussianBlur(rgb,(0,0),1.05)
    detailed = np.clip(rgb+(rgb-blurred)*(.55*S),0,1)
    rgb = blend(rgb,detailed,detail_region)

    # Luminous bloom/soft glow in bright areas + warm hint of sunlight upper left.
    glow = cv2.GaussianBlur(rgb,(0,0),13)
    luma = luminance(rgb)
    glow_alpha = np.clip((luma-.65)*.20*S,0,.045)
    rgb = blend(rgb,np.minimum(glow*1.10,1),glow_alpha)
    sun = np.exp(-(((xn-.04)/.24)**2 + ((yn-.23)/.25)**2)*1.45) * .042*S
    rgb = np.clip(rgb+sun[:,:,None]*np.array([1.0,.67,.21]),0,1)

    out = cv2.cvtColor((rgb*255+.5).astype(np.uint8),cv2.COLOR_RGB2BGR)
    cv2.imwrite(output,out,[cv2.IMWRITE_JPEG_QUALITY,97])
    print(f'Saved: {output}  |  {w} x {h} pixels')


if __name__=='__main__':
    p = argparse.ArgumentParser(description='Sunny portrait retouch (same pixel dimensions).')
    p.add_argument('input',help='Path to original JPEG')
    p.add_argument('-o','--output',default='retouched.jpg')
    p.add_argument('--strength',type=float,default=1.,help='0.0 to 1.8, default 1.0')
    args = p.parse_args()
    process(args.input,args.output,args.strength)
