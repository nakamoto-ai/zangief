# Zangief - CommuneAI Translation Subnet 


![ZANGIEF](docs/images/zangief.png)

### by Nakamoto Mining
# Documentation

[Miner Docs](docs/miners.md) | [Validator Docs](docs/validators.md) | [Discord](https://discord.com) | [Leaderboard](https://huggingface.co/spaces/ashikshaffi08/Zangief-Leaderboard)

# Purpose

> ### **The Tower of Babel**
> 
>Now the whole earth had one language and the same words. 2 And as they migrated from the east,[b] they came upon a plain in the land of Shinar and settled there. 3 And they said to one another, "Come, let us make bricks and fire them thoroughly." And they had brick for stone and bitumen for mortar. 4 Then they said, "Come, let us build ourselves a city and a tower with its top in the heavens, and let us make a name for ourselves; otherwise we shall be scattered abroad upon the face of the whole earth." 5 The LORD[c] came down to see the city and the tower, which mortals had built. 6 And the LORD said, "Look, they are one people, and they have all one language, and this is only the beginning of what they will do; nothing that they propose to do will now be impossible for them. 7 Come, let us go down and confuse their language there, so that they will not understand one another's speech." 8 So the LORD scattered them abroad from there over the face of all the earth, and they left off building the city. 9 Therefore it was called Babel, because there the LORD confused (balal) the language of all the earth, and from there the LORD scattered them abroad over the face of all the earth.
>— Genesis 11:1–9

Zangief is a subnet dedicated to language translation. The goal of the subnet is to collectively bootstrap a language translation application that supports dozens of different languages, communication styles, and specific areas of expertise. 

The actors that power the subnet are the miners and validators. The validators generate source material to be translated and pass the source material to the miners. The miners run web services that respond to the given source input with high quality translation. The miners also respond to queries that are served from an end-user application. Over time, the validators will also curate high quality translations to the source material which itself will be cleaned and compiled into a dataset. The dataset that is produced from the mining and validating activity on the subnet will be open source. This dataset can be used to train models or provide useful translations for subtitles or other online media. 

# Languages Supported

* Arabic
* Chinese
* English
* French
* German
* Hebrew
* Hindi
* Portuguese
* Russian
* Spanish
* Urdu
* Vietnamese

More to come!

# Datasets

* **[CC-100](https://huggingface.co/datasets/cc100)** - This corpus contains monolingual data for 100+ languages. This was constructed using the urls and paragraph indices provided by the CC-Net repository by processing January-December 2018 Commoncrawl snapshots.

# Scoring System

The scoring system used by the validators is a custom quality score that is adjusted over time to facilitate the highest quality translations. Translations are spot checked by human experts to ensure that the output is accurate useful.

* **Unbabel COMET** - chosen to measure how well the meaning is preserved between the source text and the translated output
* **BERTScore** - chosen to measure the semantic similarity a more granular level (token by token)

# Roadmap

* **Zangief translation app** - web app to provide high quality translations across dozens of language pairs for everyday communications
* **Zangief multilingual dataset** - open source repository of high quality translations for multilingual training and accessibility of online media
* **Zangief document translator** - web app to provide high quality translations for long-form text that maintains style and tone
* **Zangief multi-modal translator** - app that provides real-time translation of audio, visual, or text input 

## Further reading

