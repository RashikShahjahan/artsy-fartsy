const test = require('node:test');
const assert = require('node:assert/strict');
const {generating, missing, matchesFilter} = require('./comparisons.js');

test('pending outputs and completed missing images are separate filters', () => {
  const pending = {members:{base:{finished:true,image_url:'/image'},sft:{finished:false,image_url:null}},preferred:null};
  assert.equal(generating(pending),true);
  assert.equal(missing(pending),false);
  assert.equal(matchesFilter(pending,'pending'),true);
  assert.equal(matchesFilter(pending,'missing'),false);
  assert.equal(matchesFilter(pending,'unjudged'),false);
  const missingFinal = {members:{base:{finished:true,image_url:'/image'},sft:{finished:true,image_url:null}},preferred:null};
  assert.equal(generating(missingFinal),false);
  assert.equal(missing(missingFinal),true);
  assert.equal(matchesFilter(missingFinal,'unjudged'),true);
});

test('an unscored complete image set is ready for scoring, without a preferred model', () => {
  const pair={members:{base:{finished:true,image_url:'/base'},sft:{finished:true,image_url:'/sft'}},preferred:null};
  assert.equal(matchesFilter(pair,'both'),true);
  assert.equal(matchesFilter(pair,'unjudged'),true);
  assert.equal(matchesFilter(pair,'sft'),false);
});
