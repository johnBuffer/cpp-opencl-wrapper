#ifndef SOUNDPLAYER_HPP_INCLUDED
#define SOUNDPLAYER_HPP_INCLUDED

#include <SFML/Audio.hpp>
#include <string>
#include <list>

struct SoundHandler
{
    size_t maxLivingSounds;
    sf::SoundBuffer soundBuffer;
    std::list<sf::Sound> livingSounds;

    void update();
};

class SoundPlayer
{
public:
    static void      playInstanceOf(size_t soundID);
    static size_t    registerSound(const std::string& filename, size_t maxSounds = 10);
    static sf::Sound getInstanceOf(size_t soundID);

private:
    static std::vector<SoundHandler> _buffers;
};

#endif // SOUNDPLAYER_HPP_INCLUDED
