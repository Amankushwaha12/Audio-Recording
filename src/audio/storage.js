/**
 * storage.js — Separate Local Browser Storage (IndexedDB) for Original vs Processed Audio
 *
 * Uses two independent IndexedDB object stores inside `MiniAudioMasteringLabDB`:
 *   1. `originals` — Stores raw microphone recordings and saved original source clips
 *   2. `processed` — Stores mastered / processed outputs with preset & loudness metadata
 *
 * Keeping originals and processed clips in separate stores lets you download either
 * version anytime and load any stored Original (A) against any stored Processed (B)
 * take for side-by-side comparison across sessions without any backend or login.
 */

const DB_NAME = 'MiniAudioMasteringLabDB';
const DB_VERSION = 1;
export const STORE_ORIGINALS = 'originals';
export const STORE_PROCESSED = 'processed';

function openDatabase() {
  return new Promise((resolve, reject) => {
    if (typeof indexedDB === 'undefined') {
      reject(new Error('IndexedDB is not available in this browser environment.'));
      return;
    }

    const request = indexedDB.open(DB_NAME, DB_VERSION);

    request.onupgradeneeded = (event) => {
      const db = event.target.result;
      if (!db.objectStoreNames.contains(STORE_ORIGINALS)) {
        const origStore = db.createObjectStore(STORE_ORIGINALS, { keyPath: 'id' });
        origStore.createIndex('createdAt', 'createdAt', { unique: false });
      }
      if (!db.objectStoreNames.contains(STORE_PROCESSED)) {
        const procStore = db.createObjectStore(STORE_PROCESSED, { keyPath: 'id' });
        procStore.createIndex('createdAt', 'createdAt', { unique: false });
      }
    };

    request.onsuccess = () => resolve(request.result);
    request.onerror = () =>
      reject(request.error || new Error('Failed to open local audio storage database.'));
  });
}

/**
 * Lists all saved clips in either `originals` or `processed`, newest first.
 */
export async function listStoredClips(storeName) {
  const db = await openDatabase();
  return new Promise((resolve, reject) => {
    const tx = db.transaction(storeName, 'readonly');
    const store = tx.objectStore(storeName);
    const req = store.getAll();
    req.onsuccess = () => {
      const items = Array.isArray(req.result) ? req.result : [];
      items.sort((a, b) => (b.createdAt || 0) - (a.createdAt || 0));
      resolve(items);
    };
    req.onerror = () => reject(req.error);
    tx.oncomplete = () => db.close();
  });
}

/**
 * Saves a clip record (including its 16-bit WAV Blob) into the specified store.
 */
export async function saveClipToStore(storeName, record) {
  const db = await openDatabase();
  return new Promise((resolve, reject) => {
    const tx = db.transaction(storeName, 'readwrite');
    const store = tx.objectStore(storeName);
    const req = store.put(record);
    req.onsuccess = () => resolve(record);
    req.onerror = () => reject(req.error);
    tx.oncomplete = () => db.close();
  });
}

/**
 * Deletes a single clip by ID from the specified store.
 */
export async function deleteClipFromStore(storeName, id) {
  const db = await openDatabase();
  return new Promise((resolve, reject) => {
    const tx = db.transaction(storeName, 'readwrite');
    const store = tx.objectStore(storeName);
    const req = store.delete(id);
    req.onsuccess = () => resolve();
    req.onerror = () => reject(req.error);
    tx.oncomplete = () => db.close();
  });
}

/**
 * Clears all clips in the specified store (`originals` or `processed`).
 */
export async function clearStore(storeName) {
  const db = await openDatabase();
  return new Promise((resolve, reject) => {
    const tx = db.transaction(storeName, 'readwrite');
    const store = tx.objectStore(storeName);
    const req = store.clear();
    req.onsuccess = () => resolve();
    req.onerror = () => reject(req.error);
    tx.oncomplete = () => db.close();
  });
}

/**
 * Triggers a browser download for a WAV Blob.
 */
export function triggerBlobDownload(blob, filename) {
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = filename.endsWith('.wav') ? filename : `${filename}.wav`;
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}
