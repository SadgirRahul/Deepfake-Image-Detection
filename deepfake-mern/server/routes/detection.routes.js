import { Router } from 'express';
import { analyzeImage } from '../controllers/detection.controller.js';
import { uploadImage } from '../middleware/uploadImage.js';

const router = Router();

router.post('/analyze', uploadImage.single('image'), analyzeImage);

export default router;
