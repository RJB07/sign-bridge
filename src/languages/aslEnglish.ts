/** Only American Sign Language is implemented. Packs keep a second language possible later. */
export type SignLanguageId = 'ase';

export type SignLanguagePack = {
  id: SignLanguageId;
  name: string;
  outputLanguage: 'en';
  modelUrl: string;
};

export const ASL_ENGLISH_PACK: SignLanguagePack = {
  id: 'ase',
  name: 'American Sign Language',
  outputLanguage: 'en',
  modelUrl: '/models/asl-en-sequence.json',
};
