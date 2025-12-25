import express from 'express';
import { fileURLToPath } from "url";

const app = express();
const port = process.env.PORT || 8080;

const __dirname = fileURLToPath(new URL(".", import.meta.url));

app.use(express.json({ limit: '50mb' })); // Increase limit for large animation data
app.use(express.static(__dirname));

// Ensure animations directory exists
import fs from 'fs';
const animationsDir = __dirname + '/public/animations';
if (!fs.existsSync(animationsDir)) {
    fs.mkdirSync(animationsDir, { recursive: true });
}

app.post('/save-animation', (req, res) => {
    const { englishName, arabicName, frames } = req.body;

    // Validate inputs
    if (!englishName || !frames) {
        return res.status(400).send('Missing englishName or frames');
    }

    // Save Animation File (using English name)
    const filePath = `${animationsDir}/${englishName}.json`;
    fs.writeFile(filePath, JSON.stringify(frames), (err) => {
        if (err) {
            console.error(err);
            return res.status(500).send('Error saving animation file');
        }

        // Update Dictionary if Arabic name is provided
        if (arabicName) {
            const dictPath = __dirname + '/public/dictionary.json';
            let dictionary = {};

            // Load existing dictionary
            if (fs.existsSync(dictPath)) {
                try {
                    const data = fs.readFileSync(dictPath, 'utf8');
                    dictionary = JSON.parse(data);
                } catch (e) {
                    console.error("Error reading dictionary:", e);
                }
            }

            // Update dictionary
            dictionary[arabicName] = englishName;

            // Save dictionary
            fs.writeFile(dictPath, JSON.stringify(dictionary, null, 2), (err) => {
                if (err) {
                    console.error("Error saving dictionary:", err);
                    // Don't fail the whole request if dictionary fails, but log it
                }
                console.log(`Saved animation: ${englishName} (Arabic: ${arabicName})`);
                res.send('Animation and dictionary saved successfully');
            });
        } else {
            console.log(`Saved animation: ${englishName}`);
            res.send('Animation saved successfully');
        }

    });
});

// List Avatars Endpoint
app.get('/list-avatars', (req, res) => {
    const avatarsDir = __dirname + '/public/avatars';
    if (fs.existsSync(avatarsDir)) {
        fs.readdir(avatarsDir, (err, files) => {
            if (err) {
                console.error("Error scanning avatars directory:", err);
                return res.status(500).json({ error: 'Failed to scan avatars' });
            }
            // Filter for .glb or .gltf files
            const avatarFiles = files.filter(file => file.endsWith('.glb') || file.endsWith('.gltf'));
            res.json(avatarFiles);
        });
    } else {
        res.json([]); // Return empty list if dir doesn't exist
    }
});

app.get('/', (req, res) => {
    res.sendFile(__dirname + '/public/welcome.html');
});

app.get(/\/*/g, (req, res) => {
    console.log(req.path);
    res.sendFile(__dirname + '/public' + req.path);
});

app.listen(port, () => {
    console.log('Listening...');
});