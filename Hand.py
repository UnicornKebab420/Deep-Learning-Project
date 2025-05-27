from Deck import Card

class Hand:
    def __init__(self):
        self.hand = []

    def reset(self):
        self.hand = []

    def add_card(self, card:Card):
        self.hand.append(card)
    
    def get_value(self):
        value = 0
        aces = 0
        for card in self.hand:
            value += card.value
            if card.name == 'A': #can be used as 1 or 11.
                aces += 1

        while value > 21 and aces > 0:
            value -= 10
            aces -= 1
        
        return value
    
    def has_soft_ace(self):
        value = 0
        aces = 0
        for card in self.hand:
            value += card.value
            if card.name == 'A':
                aces += 1

        return (value <= 21) and (aces > 0)


    def get_first_value(self): #used if the Hand is used as a dealer.
        return self.hand[0].value

    def __str__(self):
        return "Cards:" + str(self.hand) + " | Value:" + str(self.get_value())

 