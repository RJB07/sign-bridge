export type GlossEnglish = {
  english: string;
  glosses: string;
  note: string;
};

const WH = new Set(['who', 'what', 'when', 'where', 'why', 'which', 'how']);
const TIME = new Set(['yesterday', 'today', 'tomorrow', 'now', 'morning', 'night', 'later', 'before', 'after']);
const PLACES = new Set(['store', 'home', 'outside', 'room', 'backyard', 'bedroom', 'farm', 'pool', 'closet']);
const VERBS = new Set([
  'want', 'need', 'like', 'go', 'see', 'have', 'have to', 'give', 'make', 'find', 'read', 'sleep', 'stay', 'wait',
  'ride', 'jump', 'dance', 'cry', 'think', 'say', 'talk', 'look', 'listen', 'open', 'close', 'finish', 'touch',
  'taste', 'blow', 'cut', 'drop', 'hide', 'kiss', 'nap', 'wake', 'clean', 'drink', 'eat', 'can', 'will',
  'understand', 'come',
]);
const SUBJECT: Record<string, string> = {
  me: 'I',
  i: 'I',
  you: 'you',
  we: 'we',
  us: 'we',
  he: 'he',
  she: 'she',
  it: 'it',
  yourself: 'you',
};
const POLITE = new Set(['hello', 'thank you', 'please', 'bye', 'yes', 'no', 'fine']);

type Token = { raw: string; key: string };

function keyOf(token: string): string {
  return token.trim().toLowerCase().replace(/\s+/g, ' ');
}

function capitalize(text: string): string {
  if (!text) return text;
  return text.charAt(0).toUpperCase() + text.slice(1);
}

function punctuate(text: string, question: boolean): string {
  const body = text.replace(/\s+/g, ' ').trim();
  if (!body) return '';
  const headed = capitalize(body);
  if (question) return headed.endsWith('?') ? headed : `${headed}?`;
  if (/[.!?]$/.test(headed)) return headed;
  return `${headed}.`;
}

function show(token: Token): string {
  if (/^[A-Z0-9]{2,}$/.test(token.raw)) return token.raw;
  return token.raw;
}

function subjectOf(token: Token): string {
  return SUBJECT[token.key] ?? show(token);
}

function topicComment(body: Token[]): string | null {
  if (body.length < 3 || SUBJECT[body[0].key] || WH.has(body[0].key) || !SUBJECT[body[1].key]) return null;
  const subject = subjectOf(body[1]);
  let verbAt = 2;
  let negated = false;
  if (body[2].key === 'not' && body[3] && VERBS.has(body[3].key)) {
    negated = true;
    verbAt = 3;
  } else if (!VERBS.has(body[2].key)) {
    return null;
  }
  const verb = body[verbAt].key;
  const topic = body[0];
  const rest = body.slice(verbAt + 1).map(show).join(' ');
  const place = PLACES.has(topic.key) && (verb === 'go' || verb === 'come');
  const object = place ? `to the ${show(topic)}` : show(topic);
  const helper = negated ? (subject === 'he' || subject === 'she' || subject === 'it' ? 'does not' : 'do not') : '';
  const tail = [object, rest].filter(Boolean).join(' ');
  return helper ? `${subject} ${helper} ${verb} ${tail}` : `${subject} ${verb} ${tail}`;
}

function withTime(clause: string, question: boolean, times: string[]): string {
  if (times.length === 0) return punctuate(clause, question);
  const rest = clause.startsWith('I ') || clause.startsWith('I,') ? clause : `${clause.charAt(0).toLowerCase()}${clause.slice(1)}`;
  return punctuate(`${capitalize(times.join(' '))}, ${rest}`, question);
}

/**
 * Lightweight gloss order to English. Isolated PopSign glosses are not a full
 * ASL grammar, so unknown lines stay close to the gloss order.
 */
export function glossesToEnglish(tokens: string[]): GlossEnglish {
  const glosses = tokens.map((token) => token.trim()).filter(Boolean);
  const glossLine = glosses.join(' · ');
  if (glosses.length === 0) return { english: '', glosses: '', note: '' };

  const keys = glosses.map(keyOf);
  if (keys.every((key) => POLITE.has(key))) {
    const sentence = glosses.map((token, index) => (index === 0 ? capitalize(token) : token)).join(', ');
    return { english: punctuate(sentence, false), glosses: glossLine, note: 'Greeting or short reply.' };
  }

  let body: Token[] = glosses.map((raw, index) => ({ raw, key: keys[index] }));
  const times = body.filter((token) => TIME.has(token.key)).map((token) => token.raw);
  if (times.length && times.length < body.length) body = body.filter((token) => !TIME.has(token.key));

  const please = body.length > 1 && (body[0].key === 'please' || body[body.length - 1].key === 'please');
  if (please) {
    const inner = glossesToEnglish(body.filter((token) => token.key !== 'please').map((token) => token.raw));
    const stripped = inner.english.replace(/[.!?]$/, '');
    const timed = times.length ? withTime(`${stripped}, please`, false, times) : punctuate(`${stripped}, please`, false);
    return { english: timed, glosses: glossLine, note: 'Please kept at the end.' };
  }

  if (body.length >= 2 && (body[0].key === 'my' || body[0].key === 'mine') && body[1].key === 'name') {
    const name = body.slice(2).map(show).join(' ');
    const clause = name ? `My name is ${name}` : 'My name';
    return { english: withTime(clause, false, times), glosses: glossLine, note: 'Name introduction.' };
  }
  if (body.length >= 2 && body[0].key === 'name') {
    const name = body.slice(1).map(show).join(' ');
    return { english: withTime(`My name is ${name}`, false, times), glosses: glossLine, note: 'Name introduction.' };
  }

  const topic = topicComment(body);
  if (topic) {
    return { english: withTime(topic, false, times), glosses: glossLine, note: 'Topic moved after the comment.' };
  }

  const whIndex = body.findIndex((token) => WH.has(token.key));
  if (whIndex >= 0) {
    const wh = body[whIndex].key;
    const others = body.filter((_, index) => index !== whIndex);
    const subject = others[0] && SUBJECT[others[0].key] ? subjectOf(others[0]) : '';
    const afterSubject = subject ? others.slice(1) : others;
    const verbAt = afterSubject.findIndex((token) => VERBS.has(token.key));
    let clause: string;
    if (subject && verbAt >= 0) {
      const verb = afterSubject[verbAt].key;
      const object = afterSubject.filter((_, index) => index !== verbAt).map(show).join(' ');
      const helper = subject === 'he' || subject === 'she' || subject === 'it' ? 'does' : 'do';
      clause = object ? `${wh} ${helper} ${subject} ${verb} ${object}` : `${wh} ${helper} ${subject} ${verb}`;
    } else {
      const object = others.map(show).join(' ');
      clause = object ? `${wh} is the ${object}` : wh;
    }
    return { english: withTime(clause, true, times), glosses: glossLine, note: 'Wh-word moved to the front.' };
  }

  const notAt = body.findIndex((token) => token.key === 'not');
  if (notAt > 0 && SUBJECT[body[0].key] && body[notAt + 1] && VERBS.has(body[notAt + 1].key)) {
    const subject = subjectOf(body[0]);
    const verb = body[notAt + 1].key;
    const object = body.slice(notAt + 2).map(show).join(' ');
    const helper = subject === 'he' || subject === 'she' || subject === 'it' ? 'does' : 'do';
    const clause = object ? `${subject} ${helper} not ${verb} ${object}` : `${subject} ${helper} not ${verb}`;
    return { english: withTime(clause, false, times), glosses: glossLine, note: 'Not placed before the verb.' };
  }

  if (body.length >= 2 && SUBJECT[body[0].key] && VERBS.has(body[1].key)) {
    const subject = subjectOf(body[0]);
    const verb = body[1].key;
    const objectTokens = body.slice(2);
    const object =
      (verb === 'go' || verb === 'come') && objectTokens.length === 1 && PLACES.has(objectTokens[0].key)
        ? `to the ${show(objectTokens[0])}`
        : objectTokens.map(show).join(' ');
    const clause = object ? `${subject} ${verb} ${object}` : `${subject} ${verb}`;
    const note = object.startsWith('to the ')
      ? 'Added “to the” before a place. Verb tense is unchanged.'
      : times.length
        ? 'Time word moved to the front. Verb tense is unchanged.'
        : 'Subject and verb kept in order. Verb tense is unchanged.';
    return { english: withTime(clause, false, times), glosses: glossLine, note };
  }

  const joined = body.map(show).join(' ');
  return {
    english: withTime(joined, false, times),
    glosses: glossLine,
    note: 'Word order kept. Not full ASL grammar.',
  };
}
