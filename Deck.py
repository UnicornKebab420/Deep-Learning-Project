import random
from enum import Enum 

class Suit(Enum):
    HEARTS = '♥'
    DIAMONDS = '♦'
    CLUBS = '♣'
    SPADES = '♠'

    def __str__(self):
        return self.value
    
class Card:
    def __init__(self, name:str, value:int, suit:Suit):
        self.value = value
        self.suit = suit
        self.name = name

    def __str__(self):
        return str(self.suit) + str(self.name)
    
    def __repr__(self):
        return str(self.suit) + str(self.name)
    
    def switchAce(self):
        if self.value == 11:
            self.value = 1

#This deck is a blackjack deck, 10-K will have a value of 10.
class Deck:
    def __init__(self, number_of_decks:int=1): #will use 1 deck size if nothing given.
        self.cards = self._generate_deck(number_of_decks)

    def _generate_deck(self, number_of_decks):
        face_cards = ['10', 'J', 'Q', 'K', 'A']
        deck = []

        for _ in range(0, number_of_decks):
            for suit in Suit:
                for value in range (2,11):
                    if value < 10:
                        deck.append(Card(str(value), value, suit))
                    elif value == 10:
                        for i in face_cards:
                            if i == 'A':
                                deck.append(Card(i, 11, suit))
                            else:
                                deck.append(Card(i, 10, suit))
        return deck

    def deck_size(self):
        return len(self.cards)
    
    def shuffle(self):
        random.shuffle(self.cards)

    def deal(self):
        return self.cards.pop()
    
    def __str__(self):
        return str(self.cards)