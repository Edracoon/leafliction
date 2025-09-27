from plantcv import plantcv as pcv
import rembg
from utils.error import error


class ImageProcessor:
    def __init__(self, img_path: str, transformations: list[str]):
        try:
            self.img = pcv.readimage(img_path)[0]
        except Exception as e:
            error(f"Error reading image: {e}")

        self.original = self.img.copy()
        self.gaussian = self._gray_gaussian_blur(self.img)
        self.masked = self._mask(self.img)
        self.roi, self.roi_mask = self._roi_objects(self.img, self.masked)
        self.analyzed = self._analyze_objects(self.img, self.roi_mask)
        self.plm = self._pseudolandmarks(self.img, self.roi_mask)

        # Store all transformations
        self.transformations = {
            'gaussian': self.gaussian,
            'mask': self.masked,
            'roi': self.roi,
            'analysis': self.analyzed,
            'pseudolandmarks': self.plm,
        }


    def get_transformation_title(self, transformation: str) -> str:
        """
        Return the title displayed for each transformation.
        """
        titles = {
            'gaussian': 'Gaussian Blur',
            'mask': 'Mask',
            'roi': 'ROI objects',
            'analysis': 'Analyze objects',
            'pseudolandmarks': 'Pseudolandmarks',
        }
        return titles.get(transformation, transformation.title())


    def get_transformation(self, transformation: str):
        """
        Return the transformed image corresponding.
        """
        return self.transformations[transformation]


    def _gray_gaussian_blur(self, img):
        """
        Applies Gaussian blur to an image after removing its
        background and applying a binary threshold.
        """
        img = rembg.remove(img)
        # Convert to gray scale
        gray = pcv.rgb2gray_lab(rgb_img=img,
                                channel='l')
        tresholded = pcv.threshold.binary(gray_img=gray,
                                        threshold=80,
                                        object_type='light')
        # Remove background
        tresholded = rembg.remove(tresholded, bgcolor=(0, 0, 0))
        # Apply blur
        return pcv.gaussian_blur(img=tresholded,
                                ksize=(5, 5),
                                sigma_x=0,
                                sigma_y=None)


    def _mask(self, img):
        """
        Applies a mask to an image, setting masked regions to white.
        """
        img = rembg.remove(img)
        gray = pcv.rgb2gray(rgb_img=img)
        mask = pcv.threshold.binary(gray_img=gray,
                                        threshold=80,
                                        object_type='light')
        return pcv.apply_mask(img=img,
                            mask=mask,
                            mask_color='white')


    def _roi_objects(self, img, mask):
        """
        Creates a region of interest (ROI) mask on the input
        image and highlights the ROI on the original image.
        """
        roi = pcv.roi.rectangle(img=mask,
                                x=0,
                                y=0,
                                w=img.shape[0],
                                h=img.shape[1])
        roi_mask = pcv.roi.filter(mask=pcv.threshold.binary(
                                gray_img=pcv.rgb2gray_lab(
                                    rgb_img=rembg.remove(img),
                                    channel='l'),
                                threshold=100,
                                object_type='light'),
                                roi=roi,
                                roi_type='partial')
        cpy = img.copy()
        cpy[(roi_mask != 0), 0] = 0
        cpy[(roi_mask != 0), 1] = 255
        cpy[(roi_mask != 0), 2] = 0
        return cpy, roi_mask


    def _analyze_objects(self,img, roi_mask):
        """
        Analyzes the size and shape of objects within a masked region.
        """
        return pcv.analyze.size(img=img,
                                labeled_mask=roi_mask)


    def _pseudolandmarks(self, img, roi_mask):
        """
        Adds pseudolandmark points to the input image, marking key areas.
        """
        top, bot, center_v = pcv.homology.x_axis_pseudolandmarks(img=img,
                                                                mask=roi_mask,
                                                                label='default')
        img = self._draw_pseudolandmarks(img, top, (0, 0, 255), 3)
        img = self._draw_pseudolandmarks(img, bot, (255, 0, 255), 3)
        img = self._draw_pseudolandmarks(img, center_v, (255, 0, 0), 3)
        return img


    def _draw_pseudolandmarks(self, img, coords, color, radius):
        """
        Draws pseudolandmarks as colored circles on the image
        at specified points.
        """
        for i in range(len(coords)):
            if len(coords[i]) >= 1 and len(coords[i][0]) >= 2:
                center_x = coords[i][0][1]
                center_y = coords[i][0][0]
                img = self._draw_circle(img, (center_x, center_y), color, radius)
        return img


    def _draw_circle(self, img, center, color, radius):
        """
        Draws a circle on the image at the specified center.
        """
        for x in range(img.shape[0]):
            for y in range(img.shape[1]):
                if (x - center[0]) ** 2 + (y - center[1]) ** 2 <= radius ** 2:
                    img[x, y] = color
        return img